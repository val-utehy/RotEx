"""
Configuration system for enhanced RIFT model.
"""
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class EncoderConfig:
    type: Literal["siglip", "dinov2", "evaclip"] = "dinov2"
    model_size: Literal["small", "base", "large"] = "base"
    output_dim: int = 512
    depth_aware: bool = False

@dataclass
class LossConfig:
    """
    Configuration for loss functions and weights.
    """
    recon_weight: float = 10.0          # Charbonnier Loss
    color_match_weight: float = 5.0     # ColorMatchingLoss
    day_preserve_weight: float = 3.0    # DayPreservationLoss
    moe_weight: float = 0.05            # MoE Load Balancing
    brightness_cont_weight: float = 0.1
    color_cont_weight: float = 0.1

    use_charbonnier: bool = True

    use_vgg: bool = True
    vgg_weight: float = 0.5
    # Tuning guide:
    #   0.1  → subtle sharpening, safe
    #   0.5  → noticeable edge/texture improvement  ← default

    use_lpips: bool = False
    lpips_weight: float = 0.0

    use_l1: bool = False
    use_gan: bool = False
    gan_weight: float = 0.0

    # Legacy (disabled)
    use_smooth_transition: bool = False
    smooth_weight: float = 0.0
    use_content_preservation: bool = False
    content_weight: float = 0.0
    use_brightness_preservation: bool = False
    brightness_weight: float = 0.0
    min_brightness: float = 0.0
    max_brightness: float = 0.0
    charbonnier_eps: float = 1e-4

@dataclass
class AttentionConfig:
    use_cross_attention: bool = True
    use_adain: bool = True
    use_depth_attention: bool = True
    use_multi_scale_attention: bool = False

@dataclass
class MoEConfig:
    enabled: bool = True
    num_experts: int = 4
    top_k: int = 2
    temperature: float = 0.3
    noise_std: float = 0.1
    load_balance_coef: float = 0.01

@dataclass
class SamplingConfig:
    method: Literal["euler", "heun", "consistency"] = "heun"
    num_steps: int = 30
    use_flow_matching: bool = True
    use_consistency: bool = False
    use_cfg: bool = True
    cfg_scale: float = 7.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0

@dataclass
class ModelConfig:
    in_channels: int = 16
    hidden_size: int = 1024
    num_double_layers: int = 8
    num_single_layers: int = 16
    num_heads: int = 16
    style_dim: int = 512
    use_text: bool = False
    text_dim: int = 1152
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass
class TrainingConfig:
    content_dir: str = "/workspace/ChronFLUX/rafael_ds_ver2/train/day"
    style_dir: str = "/workspace/ChronFLUX/rafael_ds_ver2/train/night"
    image_size: int = 384
    batch_size: int = 1
    grad_accum_steps: int = 4
    epochs: int = 43
    lr_max: float = 1e-7
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    save_interval_epochs: int = 1
    device: str = "cuda"
    dtype: str = "bfloat16"

def get_q1_paper_config() -> ModelConfig:
    return ModelConfig()

PRESETS = {
    "q1_paper": get_q1_paper_config,
}

def get_config(preset: str = "q1_paper") -> ModelConfig:
    return PRESETS.get(preset, get_q1_paper_config)()
