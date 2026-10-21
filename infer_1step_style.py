import os
import sys
import math
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

sys.path.insert(0, "/workspace/ChronFLUX/ChronFlux-v1")

from flux.utils.util import load_ae
from flux.chronflux import ChronFlux
from flux.modules.advanced_encoders import EncoderFactory, DepthEstimator
from config import get_config

PI = math.pi

# ===== MODEL =====
CHECKPOINT = "distill_checkpoints/best_distill.pt"

# ===== DATA (FIXED MAPPING) =====
DATA_PAIRS = [
    {
        "input":  "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/ChronFLUX-Bench/SDSD/outdoor_png/clear",
        "output": "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/DS-BenchFinal/SDSD/outdoor_png/dark_gen",
    },
    {
        "input":  "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/ChronFLUX-Bench/SDSD/indoor_png/clear",
        "output": "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/DS-BenchFinal/SDSD/indoor_png/dark_gen",
    },
    {
        "input":  "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/ChronFLUX-Bench/SMID/clear",
        "output": "/workspace/ChronFLUX/ChronFlux-v1/EvaluationDS/DS-BenchFinal/SMID/dark_gen",
    },
]

# ===== STYLE =====
FIXED_STYLE_PATH = "/workspace/ChronFLUX/rafael_ds_ver2/train/night/frame_000002.jpg"

# ===== PARAMS =====
HOURS = list(range(1, 25))

IMAGE_SIZE = 384
DEPTH_SIZE = 48
DEVICE     = "cuda"
DTYPE      = torch.bfloat16


def load_model(ckpt_path, model_config, device, dtype):
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

    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)

    print(f"[INFO] Loaded checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def infer_1step(model, ae, content_lat, depth_small, style_vec, phi, device, dtype):
    t = torch.tensor([0.0], device=device, dtype=dtype)
    z = torch.randn_like(content_lat)

    v, _ = model(z, content_lat, depth_small, style_vec, t, phi)
    z_final = (z + v * 1.0).clamp(-8, 8)

    return ae.decode(z_final).clamp(-1, 1)


def save_tensor(tensor, path):
    img = tensor.squeeze(0).cpu().float()
    img = ((img + 1.0) / 2.0).clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def main():
    model_config = get_config("q1_paper")

    print("[INFO] Loading AutoEncoder...")
    ae = load_ae(DEVICE).to(DEVICE).to(DTYPE).eval().requires_grad_(False)

    print("[INFO] Loading Style Encoder (DINOv2)...")
    style_enc = EncoderFactory.create_encoder(
        "dinov2", DEVICE, 512, False, model_size="base"
    ).to(DEVICE).to(DTYPE).eval().requires_grad_(False)

    print("[INFO] Loading Depth Estimator...")
    depth_est = DepthEstimator(DEVICE)

    print("[INFO] Loading distilled model...")
    model = load_model(CHECKPOINT, model_config, DEVICE, DTYPE)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    print("[INFO] Encoding style image...")
    style_t = transform(
        Image.open(FIXED_STYLE_PATH).convert("RGB")
    ).unsqueeze(0).to(DEVICE).to(DTYPE)

    with torch.no_grad():
        style_vec = style_enc(style_t, return_spatial=True)

    print(f"[INFO] Style vector shape: {style_vec.shape}")

    # ===== LOOP DATA PAIRS =====
    for pair in DATA_PAIRS:
        content_dir = pair["input"]
        output_dir  = pair["output"]

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(content_dir):
            print(f"[WARNING] Missing folder: {content_dir}")
            continue

        content_files = sorted([
            f for f in os.listdir(content_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        if not content_files:
            print(f"[WARNING] No images in: {content_dir}")
            continue

        print(f"\n[DATASET]")
        print(f"[INPUT ] {content_dir}")
        print(f"[OUTPUT] {output_dir}")
        print(f"[INFO  ] {len(content_files)} images")

        for fname in tqdm(content_files, desc=os.path.basename(content_dir)):
            stem = os.path.splitext(fname)[0]

            try:
                content_tensor = transform(
                    Image.open(os.path.join(content_dir, fname)).convert("RGB")
                ).unsqueeze(0).to(DEVICE).to(DTYPE)

                with torch.no_grad():
                    depth_full = depth_est.predict(
                        content_tensor.float(), IMAGE_SIZE
                    ).to(DTYPE)

                    depth_small = F.interpolate(
                        depth_full,
                        (DEPTH_SIZE, DEPTH_SIZE),
                        mode="nearest"
                    )

                    content_lat = ae.encode(content_tensor)

                # ===== LOOP HOURS =====
                for h in HOURS:
                    phi = torch.tensor(
                        [h * (2 * PI / 24.0)],
                        device=DEVICE,
                        dtype=DTYPE
                    )

                    out = infer_1step(
                        model,
                        ae,
                        content_lat,
                        depth_small,
                        style_vec,
                        phi,
                        DEVICE,
                        DTYPE
                    )

                    save_path = os.path.join(
                        output_dir,
                        f"{stem}_h{h:02d}.png"
                    )

                    save_tensor(out, save_path)

            except Exception as e:
                print(f"\n[ERROR] {fname}: {e}")
                import traceback
                traceback.print_exc()

        print(f"[DONE] {content_dir}: {len(content_files)*len(HOURS)} files")

    print("\n🎉 ALL DATASETS DONE")


if __name__ == "__main__":
    main()
