import os
import sys
import argparse
import math
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

try:
    from moviepy import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

sys.path.insert(0, "/workspace/ChronFLUX")

from flux.utils.util import load_ae
from flux.chronflux import ChronFlux
from flux.modules.advanced_encoders import EncoderFactory, DepthEstimator
from config import get_config

PI = math.pi


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
    model.requires_grad_(False)
    return model


def get_cosine_schedule(num_steps):
    ts = []
    for i in range(num_steps):
        t_val = 1.0 - math.cos((i / num_steps) * math.pi / 2.0)
        ts.append(t_val)
    return ts


def get_safe_resolution(width, height, max_height=720):
    if height <= max_height and width % 16 == 0 and height % 16 == 0:
        return width, height
    ratio = max_height / height
    new_width = int(width * ratio)
    new_height = max_height
    if new_width % 16 != 0:
        new_width = new_width - (new_width % 16)
    if new_height % 16 != 0:
        new_height = new_height - (new_height % 16)
    return new_width, new_height


def tensor_to_numpy_bgr(tensor):
    img = tensor.clamp(-1, 1)
    img = (img + 1.0) / 2.0
    img = img.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def inference_single_frame(
        model, ae, depth_est,
        frame_tensor,
        style_vec_cached,
        phi, device, dtype,
        denoise_steps, depth_size,
        base_noise
):
    with torch.no_grad():
        image_size = frame_tensor.shape[-1]

        depth_full = depth_est.predict(frame_tensor.float(), image_size).to(dtype)
        depth_small = F.interpolate(depth_full, size=(depth_size, depth_size), mode="nearest")

        # content_gray = transforms.functional.rgb_to_grayscale(frame_tensor, num_output_channels=3)
        # content_lat = ae.encode(content_gray)
        content_lat = ae.encode(frame_tensor)

        style_vec = style_vec_cached

        t_schedule = get_cosine_schedule(denoise_steps)

        z = base_noise.clone()

        for i in range(denoise_steps):
            t_val = t_schedule[i]
            if i < denoise_steps - 1:
                dt = t_schedule[i + 1] - t_schedule[i]
            else:
                dt = 1.0 - t_schedule[i]

            t = torch.tensor([t_val], device=device, dtype=dtype)

            v1, _ = model(z, content_lat, depth_small, style_vec, t, phi)
            z_mid = (z + v1 * dt).clamp(-8, 8)

            if i < denoise_steps - 1:
                t_next_val = t_schedule[i + 1]
            else:
                t_next_val = 1.0
            t_next = torch.tensor([t_next_val], device=device, dtype=dtype)
            v2, _ = model(z_mid, content_lat, depth_small, style_vec, t_next, phi)

            z = z + (v1 + v2) * 0.5 * dt
            z = z.clamp(-8, 8)

        z = z.clamp(-6, 6)
        out = ae.decode(z).clamp(-1, 1)
        return out


def main():
    parser = argparse.ArgumentParser(description="ChronFlux Video Inference - Day/Night Cycle (Fast)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--style_path", type=str, required=True, help="Ảnh style ban đêm")
    parser.add_argument("--output_dir", type=str, default="/workspace/ChronFLUX/video_outputs")
    parser.add_argument("--denoise_steps", type=int, default=50)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--depth_size", type=int, default=48)
    parser.add_argument("--max_height", type=int, default=384)
    parser.add_argument("--hour_start", type=float, default=0.0)
    parser.add_argument("--hour_end", type=float, default=24.0)
    parser.add_argument("--cycle", action="store_true")
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading config...")
    model_config = get_config("q1_paper")

    print("[INFO] Loading AutoEncoder...")
    ae = load_ae(device).to(device).to(dtype).eval().requires_grad_(False)

    print("[INFO] Loading Style Encoder (DINOv2)...")
    style_enc = EncoderFactory.create_encoder(
        "dinov2", device, 512, False, model_size="base"
    ).to(device).to(dtype).eval().requires_grad_(False)

    print("[INFO] Loading Depth Estimator...")
    depth_est = DepthEstimator(device)

    print(f"[INFO] Loading ChronFlux from {args.checkpoint}...")
    model = load_model(args.checkpoint, model_config, device, dtype)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    print("[INFO] Loading & encoding style image (cached for all frames)...")
    style_pil = Image.open(args.style_path).convert("RGB")
    style_tensor = transform(style_pil).unsqueeze(0).to(device).to(dtype)
    with torch.no_grad():
        style_vec_cached = style_enc(style_tensor, return_spatial=True)
    print(f"[INFO] Style vector cached: {style_vec_cached.shape}")
    # ─────────────────────────────────────────────────────────────────────────

    print("[INFO] Opening video...")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video: {args.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc_w, proc_h = get_safe_resolution(orig_w, orig_h, args.max_height)
    print(f"[INFO] Video: {orig_w}x{orig_h} → resize {proc_w}x{proc_h}, {fps:.2f} FPS, {total_frames} frames")
    print(f"[INFO] Hour range: {args.hour_start}h → {args.hour_end}h, denoise_steps={args.denoise_steps} (Heun)")
    print(f"[INFO] Speedup: style_enc skipped for {total_frames - 1} frames")

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    temp_path = os.path.join(args.output_dir, f"{video_name}_temp.mp4")
    output_path = os.path.join(args.output_dir, f"{video_name}_chronflux.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (proc_w, proc_h))

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Processing frames")

    base_noise = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        frame_tensor_inf = transform(frame_pil).unsqueeze(0).to(device).to(dtype)

        if base_noise is None:
            with torch.no_grad():
                content_gray = transforms.functional.rgb_to_grayscale(frame_tensor_inf, num_output_channels=3)
                first_lat = ae.encode(content_gray)
                base_noise = torch.randn_like(first_lat)
            print(f"[INFO] Base noise initialized: {base_noise.shape}")

        if args.cycle:
            progress = (frame_idx % total_frames) / max(total_frames - 1, 1)
            if (frame_idx // total_frames) % 2 == 0:
                hour = args.hour_start + progress * (args.hour_end - args.hour_start)
            else:
                hour = args.hour_end - progress * (args.hour_end - args.hour_start)
        else:
            progress = frame_idx / max(total_frames - 1, 1)
            hour = args.hour_start + progress * (args.hour_end - args.hour_start)

        phi_val = hour * (2 * PI / 24.0)
        phi = torch.tensor([phi_val], device=device, dtype=dtype)

        out = inference_single_frame(
            model=model,
            ae=ae,
            depth_est=depth_est,
            frame_tensor=frame_tensor_inf,
            style_vec_cached=style_vec_cached,
            phi=phi,
            device=device,
            dtype=dtype,
            denoise_steps=args.denoise_steps,
            depth_size=args.depth_size,
            base_noise=base_noise
        )

        out_bgr = tensor_to_numpy_bgr(out)
        out_bgr_resized = cv2.resize(out_bgr, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        writer.write(out_bgr_resized)

        frame_idx += 1
        pbar.update(1)
        pbar.set_postfix({"hour": f"{hour:.1f}h", "phi": f"{phi_val:.3f}"})

    cap.release()
    writer.release()
    pbar.close()
    print(f"[INFO] Finished processing {frame_idx} frames.")

    if not args.no_audio and MOVIEPY_AVAILABLE:
        print("[INFO] Adding audio from original video...")
        try:
            original_clip = VideoFileClip(args.video_path)
            generated_clip = VideoFileClip(temp_path)
            if original_clip.audio is not None:
                final_clip = generated_clip.set_audio(original_clip.audio)
                final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger="bar")
                original_clip.close()
                generated_clip.close()
                os.remove(temp_path)
                print("[INFO] Audio added successfully.")
            else:
                original_clip.close()
                generated_clip.close()
                os.rename(temp_path, output_path)
                print("[INFO] No audio track found in original video.")
        except Exception as e:
            print(f"[WARN] Could not add audio: {e}")
            os.rename(temp_path, output_path)
    else:
        if not MOVIEPY_AVAILABLE and not args.no_audio:
            print("[WARN] moviepy not installed, skipping audio.")
        os.rename(temp_path, output_path)

    print(f"\n[DONE] Output saved to: {output_path}")


if __name__ == "__main__":
    main()
