import os
import sys
import argparse
import math
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

sys.path.insert(0, "/workspace/RIFT-9226")

from flux.utils.util import load_ae
from flux.chronflux import ChronFlux
from flux.modules.advanced_encoders import EncoderFactory, DepthEstimator
from config import get_config

PI = np.pi


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def add_label(tensor_img, text):
    img = (tensor_img.clamp(-1, 1) + 1) / 2
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    x, y = 6, 6
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        draw.text((x + dx, y + dy), text, fill="black", font=font)
    draw.text((x, y), text, fill="white", font=font)
    out = np.array(pil_img).astype(np.float32) / 255.0
    out = torch.from_numpy(out).permute(2, 0, 1).unsqueeze(0)
    return out * 2 - 1


def load_model(checkpoint_path, model_config, device, dtype):
    model = ChronFlux(
        in_channels=16,
        hidden_size=1024,
        num_double_layers=8,
        num_single_layers=16,
        num_heads=16,
        style_dim=512,
        use_moe=True,
        use_depth_attn=True,
        use_adain=True,
        moe_params={"num_experts": model_config.moe.num_experts}
    ).to(device).to(dtype)

    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def get_cosine_schedule(num_steps):
    return [1.0 - math.cos((i / num_steps) * math.pi / 2.0) for i in range(num_steps)]


# ─────────────────────────────────────────────────────────────────────────────
#  Core denoising loop  (shared by single-seed and multi-seed paths)
# ─────────────────────────────────────────────────────────────────────────────

def denoise(
        model, z_init, content_lat, depth_small, style_vec,
        phi, device, dtype, denoise_steps,
        latent_clamp=20.0,          # clamp during ODE integration
        latent_clamp_final=15.0,    # tighter clamp right before decode
):
    """
    Heun 2nd-order rectified-flow ODE solver.

    Returns the final latent (NOT yet decoded).
    Caller is responsible for ae.decode(z).
    """
    t_schedule = get_cosine_schedule(denoise_steps)
    z = z_init.clone()

    for i in range(denoise_steps):
        t_val = t_schedule[i]
        dt    = (t_schedule[i + 1] - t_schedule[i]) if i < denoise_steps - 1 else (1.0 - t_schedule[i])

        t_cur = torch.tensor([t_val], device=device, dtype=dtype)

        # ── Euler predictor ─────────────────────────────────────────────────
        v1, _ = model(z, content_lat, depth_small, style_vec, t_cur, phi)
        z_mid  = (z + v1 * dt).clamp(-latent_clamp, latent_clamp)

        # ── Heun corrector ───────────────────────────────────────────────────
        t_next_val = t_schedule[i + 1] if i < denoise_steps - 1 else 1.0
        t_next = torch.tensor([t_next_val], device=device, dtype=dtype)
        v2, _ = model(z_mid, content_lat, depth_small, style_vec, t_next, phi)

        z = z + (v1 + v2) * 0.5 * dt
        z = z.clamp(-latent_clamp, latent_clamp)               # keep ODE stable

    # ── Final pre-decode clamp ───────────────────────────────────────────────
    z = z.clamp(-latent_clamp_final, latent_clamp_final)
    return z


# ─────────────────────────────────────────────────────────────────────────────
#  Per-image inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
        model, ae, style_enc, depth_est,
        content_path, style_path,
        device, dtype,
        denoise_steps, image_size, depth_size,
        noise_alpha=0.92,      # [DEPRECATED] kept for signature compatibility
        num_seeds=1,           # >1 → average over multiple trajectories
        hours=None,            # list of hours to render; default = 0..24
):
    """
    Returns a strip tensor:
        [content | style | depth | hour_0 | hour_1 | ... | hour_24]
    """
    if hours is None:
        hours = list(range(25))      # 0 h … 24 h

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    content_tensor = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device).to(dtype)
    style_tensor   = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device).to(dtype)

    with torch.no_grad():
        # ── Encode once ─────────────────────────────────────────────────────
        depth_full  = depth_est.predict(content_tensor.float(), image_size).to(dtype)
        depth_small = F.interpolate(depth_full, size=(depth_size, depth_size), mode="nearest")

        content_lat = ae.encode(content_tensor)

        style_vec = style_enc(style_tensor, return_spatial=True)

        depth_vis = (depth_full.repeat(1, 3, 1, 1) - 0.5) / 0.5

        generated = []
        for h in hours:
            phi_val = h * (2 * PI / 24.0)
            phi     = torch.tensor([phi_val], device=device, dtype=dtype)

            if num_seeds == 1:
                # ── Single trajectory ────────────────────────────────────────
                z_init = torch.randn_like(content_lat)

                z_final = denoise(
                    model, z_init, content_lat, depth_small, style_vec,
                    phi, device, dtype, denoise_steps
                )
                out = ae.decode(z_final).clamp(-1, 1)

            else:
                decoded_sum = torch.zeros(1, 3, image_size, image_size, device=device, dtype=torch.float32)
                for _ in range(num_seeds):
                    z_init = torch.randn_like(content_lat)

                    z_final = denoise(
                        model, z_init, content_lat, depth_small, style_vec,
                        phi, device, dtype, denoise_steps
                    )
                    decoded_sum += ae.decode(z_final).clamp(-1, 1).float()
                out = (decoded_sum / num_seeds).clamp(-1, 1)

            generated.append(add_label(out.float(), f"{h}h").to(device))

    content_labeled = add_label(content_tensor.float(), "Content").to(device)
    style_labeled   = add_label(style_tensor.float(),   "Style").to(device)
    depth_labeled   = add_label(depth_vis.float(),      "Depth").to(device)

    strip = torch.cat([content_labeled, style_labeled, depth_labeled] + generated, dim=3)
    return strip


# ─────────────────────────────────────────────────────────────────────────────
#  CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ChronFlux 24-hour inference")
    parser.add_argument("--checkpoint",    type=str,   default="checkpoints/full/best.pt")
    parser.add_argument("--content_dir",   type=str,   default="/workspace/RIFT/rafael_ds_ver2/img_test/cnt")
    parser.add_argument("--style_dir",     type=str,   default="/workspace/RIFT/rafael_ds_ver2/img_test/sty")
    parser.add_argument("--output_dir",    type=str,   default="/workspace/RIFT-9226/inference_outputs")
    parser.add_argument("--denoise_steps", type=int,   default=50,
                        help="More steps = better quality. 50 is a good default; try 75 for maximum quality.")
    parser.add_argument("--image_size",    type=int,   default=384)
    parser.add_argument("--depth_size",    type=int,   default=48)
    parser.add_argument("--noise_alpha",   type=float, default=0.85,
                        help="[DEPRECATED] No longer used. Model requires pure noise init.")
    parser.add_argument("--num_seeds",     type=int,   default=1,
                        help="Average over N random seeds (slow but reduces noise artifacts).")
    parser.add_argument("--device",        type=str,   default="cuda")
    parser.add_argument("--dtype",         type=str,   default="bfloat16")
    args = parser.parse_args()

    device = args.device
    dtype  = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Loading config...")
    model_config = get_config("q1_paper")

    print(f"[INFO] Loading AutoEncoder...")
    ae = load_ae(device).to(device).to(dtype).eval().requires_grad_(False)

    print(f"[INFO] Loading Style Encoder (DINOv2)...")
    style_enc = EncoderFactory.create_encoder(
        "dinov2", device, 512, False, model_size="base"
    ).to(device).to(dtype).eval().requires_grad_(False)

    print(f"[INFO] Loading Depth Estimator...")
    depth_est = DepthEstimator(device)

    print(f"[INFO] Loading ChronFlux from {args.checkpoint}...")
    model = load_model(args.checkpoint, model_config, device, dtype)
    model.requires_grad_(False)

    print(f"[INFO] denoise_steps={args.denoise_steps} | num_seeds={args.num_seeds} | Pure Noise Init: ENABLED")

    content_files = sorted([
        f for f in os.listdir(args.content_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    style_files = sorted([
        f for f in os.listdir(args.style_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not content_files:
        print("[ERROR] No content images found.")
        return
    if not style_files:
        print("[ERROR] No style images found.")
        return

    print(f"[INFO] Found {len(content_files)} content images and {len(style_files)} style images.")

    for content_file in tqdm(content_files, desc="Content images"):
        content_path = os.path.join(args.content_dir, content_file)
        content_stem = os.path.splitext(content_file)[0]

        for style_file in style_files:
            style_path = os.path.join(args.style_dir, style_file)
            style_stem = os.path.splitext(style_file)[0]

            out_name = f"{content_stem}__sty_{style_stem}.png"
            out_path = os.path.join(args.output_dir, out_name)

            if os.path.exists(out_path):
                print(f"[SKIP] {out_name} already exists.")
                continue

            try:
                strip = run_inference(
                    model=model,
                    ae=ae,
                    style_enc=style_enc,
                    depth_est=depth_est,
                    content_path=content_path,
                    style_path=style_path,
                    device=device,
                    dtype=dtype,
                    denoise_steps=args.denoise_steps,
                    image_size=args.image_size,
                    depth_size=args.depth_size,
                    noise_alpha=args.noise_alpha,
                    num_seeds=args.num_seeds,
                )

                save_image(strip, out_path, normalize=True, value_range=(-1, 1))
                print(f"[SAVED] {out_path}")

            except Exception as e:
                print(f"[ERROR] Failed on {content_file} x {style_file}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n[DONE] All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
