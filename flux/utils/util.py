import os
from dataclasses import dataclass
from pathlib import Path
import torch
from einops import rearrange
from PIL import Image
from safetensors.torch import load_file as load_sft
from huggingface_hub import hf_hub_download
from flux.modules.autoencoder import AutoEncoder, AutoEncoderParams

CHECKPOINTS_DIR = Path("checkpoints")
CHECKPOINTS_DIR.mkdir(exist_ok=True)


@dataclass
class VAEConfig:
    repo_id: str
    repo_ae: str
    ae_params: AutoEncoderParams


flux_vae_config = VAEConfig(
    repo_id="black-forest-labs/FLUX.1-dev",
    repo_ae="ae.safetensors",
    ae_params=AutoEncoderParams(
        resolution=64,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
    ),
)


def get_checkpoint_path(filename: str, repo_id: str) -> Path:
    local_path = CHECKPOINTS_DIR / filename
    if not local_path.exists():
        print(f"Downloading {filename} from {repo_id}...")
        try:
            downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=CHECKPOINTS_DIR)
            return Path(downloaded_path)
        except Exception as e:
            print(f"Failed to download: {e}")
    return local_path


def load_ae(device: str | torch.device = "cuda") -> AutoEncoder:
    print("Init AE")
    ckpt_path = get_checkpoint_path(flux_vae_config.repo_ae, flux_vae_config.repo_id)

    if ckpt_path.exists():
        print(f"Loading AE checkpoint: {ckpt_path}")
        with torch.device("meta"):
            ae = AutoEncoder(flux_vae_config.ae_params)
        sd = load_sft(str(ckpt_path), device=str(device))
        ae.load_state_dict(sd, strict=False, assign=True)
    else:
        raise RuntimeError("AE Checkpoint is missing! Model cannot train with random VAE.")

    return ae


def save_image(x: torch.Tensor, path: str):
    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
    img.save(path, quality=95, subsampling=0)


def print_load_warning(missing: list[str], unexpected: list[str]):
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Parameters missing: {len(missing)}. Unexpected: {len(unexpected)}")