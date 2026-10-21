"""
Adaptive Normalization Modules for ChronFlux
Includes AdaIN (Adaptive Instance Normalization) and AdaLN (Adaptive Layer Normalization)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


NIGHT_SKY_INTENSITY = -0.95
WARM_BIAS_RGB = [0.2, 0.0, -0.2]


class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    Used for generating pseudo-style targets during training with extensive augmentation.

    Reference: Huang & Belongie, "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, feat):
        size = feat.size()
        N, C = size[:2]
        feat_view = feat.view(N, C, -1)

        feat_std, feat_mean = torch.std_mean(feat_view, dim=2, unbiased=False)

        feat_std = (feat_std + self.eps).view(N, C, 1, 1)
        feat_mean = feat_mean.view(N, C, 1, 1)

        return feat_mean, feat_std

    def forward(self, content, style):
        """
        Apply AdaIN: transfers style statistics to content

        Args:
            content: Content features [B, C, H, W]
            style: Style features [B, C, H, W]

        Returns:
            Stylized content with style's mean and std
        """
        size = content.size()
        content_mean, content_std = self.calc_mean_std(content)
        style_mean, style_std = self.calc_mean_std(style)

        normalized = (content - content_mean) / content_std
        stylized = normalized * style_std + style_mean

        return stylized


class AdaINTransfer(nn.Module):
    """
    Extended AdaIN for pseudo-style target generation with augmentation support.
    Used specifically for training pipeline to create robust style targets.
    """
    def __init__(self, eps=1e-5):
        super().__init__()
        self.adain = AdaIN(eps=eps)

        self.register_buffer(
            'warm_bias_template',
            torch.tensor(WARM_BIAS_RGB, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def adaptive_night_transfer(self, content, style, sky_mask=None):
        """
        Transfer night style to content with optional sky masking

        Args:
            content: Day images [B, C, H, W]
            style: Night images [B, C, H, W]
            sky_mask: Binary mask for sky regions [B, 1, H, W]

        Returns:
            Night-styled content
        """
        target = self.adain(content, style)

        if sky_mask is not None:
            target = target * (1 - sky_mask) + (NIGHT_SKY_INTENSITY * sky_mask)

        return target.clamp(-0.95, 0.95)

    def adaptive_sunset_transfer(self, content, style_night, sky_mask=None):
        """
        Create sunset style by channel manipulation of night style

        Args:
            content: Day images [B, C, H, W]
            style_night: Night images for base style [B, C, H, W]
            sky_mask: Binary mask for sky regions [B, 1, H, W]

        Returns:
            Sunset-styled content with warm tones
        """
        content_mean, content_std = self.adain.calc_mean_std(content)
        style_mean, style_std = self.adain.calc_mean_std(style_night)

        mu_swap = style_mean.clone()
        mu_swap[:, 0] = style_mean[:, 2]
        mu_swap[:, 2] = style_mean[:, 0]

        std_swap = style_std.clone()
        std_swap[:, 0] = style_std[:, 2]
        std_swap[:, 2] = style_std[:, 0]

        B = content.shape[0]
        warm_bias = self.warm_bias_template.to(content.dtype).expand(B, 3, 1, 1)

        mu_final = mu_swap * 0.7 + warm_bias * 0.3
        std_final = std_swap * 1.4

        target = (content - content_mean) / content_std * std_final + mu_final

        if sky_mask is not None:
            sky_color = mu_final.view(B, 3, 1, 1)
            target = target * (1 - sky_mask) + (target * 0.4 + sky_color * 0.6) * sky_mask

        return target.clamp(-0.95, 0.95)


class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) Modulation.
    Generates scale, shift, and gate parameters for layer-wise adaptive normalization.
    Provides better control over style consistency and layer-wise adjustments.
    """
    def __init__(self, hidden_size, components=6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, components * hidden_size, bias=True)
        self.components = components

        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, vec):
        """
        Generate modulation parameters

        Args:
            vec: Conditioning vector [B, hidden_size]

        Returns:
            Tuple of (shift, scale, gate) parameters for each component
        """
        chunk = self.linear(self.silu(vec))
        return chunk.chunk(self.components, dim=1)


class AdaLNBlock(nn.Module):
    """
    Base AdaLN normalization block with scale, shift, and gate.
    Can be used as a building block for various architectures.
    """
    def __init__(self, hidden_size, use_gate=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.use_gate = use_gate

    def forward(self, x, scale, shift, gate=None):
        """
        Apply AdaLN: normalize, then scale and shift, optionally gate

        Args:
            x: Input features [B, L, D]
            scale: Scale parameter [B, D]
            shift: Shift parameter [B, D]
            gate: Optional gate parameter [B, D]

        Returns:
            Modulated features
        """
        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        if self.use_gate and gate is not None:
            x_mod = gate.unsqueeze(1) * x_mod

        return x_mod


class AdaLNZero(nn.Module):
    """
    AdaLN-Zero: AdaLN with zero-initialized output scaling.
    Particularly useful for residual connections and stable training.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale = nn.Parameter(torch.zeros(hidden_size))
        self.shift = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, modulation_scale, modulation_shift):
        """
        Apply AdaLN with zero-initialization

        Args:
            x: Input features [B, L, D]
            modulation_scale: Additional scale from conditioning [B, D]
            modulation_shift: Additional shift from conditioning [B, D]

        Returns:
            Modulated features
        """
        x_norm = self.norm(x)
        x_mod = x_norm * (self.scale + modulation_scale.unsqueeze(1)) + (self.shift + modulation_shift.unsqueeze(1))
        return x_mod


class StyleAdaLN(nn.Module):
    """
    Style-aware AdaLN that combines style and conditioning vectors.
    Particularly useful for style transfer with additional controls.
    """
    def __init__(self, hidden_size, style_dim):
        super().__init__()
        self.style_proj = nn.Linear(style_dim, hidden_size)
        self.adaln = AdaLNModulation(hidden_size, components=2)
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, style_vec, cond_vec=None):
        """
        Apply style-aware normalization

        Args:
            x: Input features [B, L, D]
            style_vec: Style vector [B, style_dim]
            cond_vec: Optional additional conditioning [B, D]

        Returns:
            Style-modulated features
        """
        style_emb = self.style_proj(style_vec)

        if cond_vec is not None:
            combined = style_emb + cond_vec
        else:
            combined = style_emb

        scale, shift = self.adaln(combined)

        x_norm = self.norm(x)
        x_mod = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        return x_mod


__all__ = [
    'AdaIN',
    'AdaINTransfer',
    'AdaLNModulation',
    'AdaLNBlock',
    'AdaLNZero',
    'StyleAdaLN',
]