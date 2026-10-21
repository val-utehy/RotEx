"""
Adaptive Flux Blocks with enhanced AdaLN integration.
These blocks serve as building blocks for the SwiftFlux architecture with
advanced style transfer control and optional MoE-compatible components.
Author: https://github.com/datnguyen-tien204
"""
import torch
from torch import nn
from einops import rearrange
from flux.modules.normalization import AdaLNModulation
from flux.utils.math import attention, apply_rope
from flux.utils.logger_utils import ConsoleLogger


# Constants
SWIGLU_HIDDEN_RATIO = 2 / 3  # Hidden dimension ratio for SwiGLU


class QKNorm(nn.Module):
    """Query-Key normalization for stable attention"""
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.key_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q), self.key_norm(k)


class SwiGLU(nn.Module):
    """SwiGLU activation for efficient FFN"""
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features * SWIGLU_HIDDEN_RATIO)

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.silu(x1) * x2
        return self.w3(hidden)


class AdaptiveFluxBlock(nn.Module):
    """
    Base Adaptive Flux Block with AdaLN integration.
    Serves as foundation for both double-stream and single-stream variants.
    
    Features:
    - AdaLN for layer-wise style and conditioning control
    - QKNorm for attention stability
    - SwiGLU for efficient feed-forward
    - Optional sparse activation support for MoE compatibility
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_sparse=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_sparse = use_sparse
        # Note: adaLN should be initialized in subclasses
        
    def apply_adaptive_norm(self, x, scale, shift):
        """Apply AdaLN normalization with scale and shift"""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaptiveDoubleStreamBlock(AdaptiveFluxBlock):
    """
    Adaptive Double-Stream Block with enhanced AdaLN.
    Processes image and style streams separately with cross-attention.
    
    Improvements:
    - Dedicated AdaLN for both streams
    - Enhanced normalization for better style consistency
    - Support for sparse activation patterns
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_sparse=False):
        super().__init__(hidden_size, num_heads, mlp_ratio, use_sparse)
        
        # Image stream components
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        self.img_qk_norm = QKNorm(self.head_dim)
        
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        
        # Style stream components
        self.sty_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sty_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.sty_proj = nn.Linear(hidden_size, hidden_size)
        self.sty_qk_norm = QKNorm(self.head_dim)
        
        self.sty_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sty_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        
        # AdaLN with 12 components (6 for img, 6 for style)
        self.adaLN = AdaLNModulation(hidden_size, components=12)
    
    def forward(self, img, sty, vec, pe):
        """
        Forward pass with dual-stream processing
        
        Args:
            img: Image features [B, L_img, D]
            sty: Style features [B, L_sty, D]
            vec: Conditioning vector [B, D]
            pe: Position embeddings
        
        Returns:
            Updated (img, sty) features
        """
        # Generate all modulation parameters
        params = self.adaLN(vec)
        (shift_i_a, scale_i_a, gate_i_a, shift_i_m, scale_i_m, gate_i_m,
         shift_s_a, scale_s_a, gate_s_a, shift_s_m, scale_s_m, gate_s_m) = params
        
        # Image stream - attention
        img_mod = self.apply_adaptive_norm(self.img_norm1(img), scale_i_a, shift_i_a)
        sty_mod = self.apply_adaptive_norm(self.sty_norm1(sty), scale_s_a, shift_s_a)
        
        qkv_i = self.img_qkv(img_mod)
        qkv_s = self.sty_qkv(sty_mod)
        
        q_i, k_i, v_i = rearrange(qkv_i, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q_s, k_s, v_s = rearrange(qkv_s, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        
        q_i, k_i = self.img_qk_norm(q_i, k_i)
        q_s, k_s = self.sty_qk_norm(q_s, k_s)
        
        # Preserve dtype for Flash Attention
        dtype = img.dtype
        q_i, k_i, v_i = q_i.to(dtype), k_i.to(dtype), v_i.to(dtype)
        q_s, k_s, v_s = q_s.to(dtype), k_s.to(dtype), v_s.to(dtype)
        
        # Apply RoPE if position embeddings provided
        if pe is not None:
            pe = pe.transpose(1, 2)
            pe_s = pe[:, :1, ...]
            pe_i = pe[:, 1:, ...]
            
            q_i, k_i = apply_rope(q_i, k_i, pe_i)
            q_s, k_s = apply_rope(q_s, k_s, pe_s)
            
            q_i, k_i = q_i.to(dtype), k_i.to(dtype)
            q_s, k_s = q_s.to(dtype), k_s.to(dtype)
        
        # Cross-attention between streams
        full_k = torch.cat([k_s, k_i], dim=1)
        full_v = torch.cat([v_s, v_i], dim=1)
        
        img_attn_out = self.img_proj(attention(q_i, full_k, full_v))
        sty_attn_out = self.sty_proj(attention(q_s, full_k, full_v))
        
        # Apply gating and residual
        img = img + gate_i_a.unsqueeze(1) * img_attn_out
        sty = sty + gate_s_a.unsqueeze(1) * sty_attn_out
        
        # MLP blocks with AdaLN
        img_mod = self.apply_adaptive_norm(self.img_norm2(img), scale_i_m, shift_i_m)
        img = img + gate_i_m.unsqueeze(1) * self.img_mlp(img_mod)
        
        sty_mod = self.apply_adaptive_norm(self.sty_norm2(sty), scale_s_m, shift_s_m)
        sty = sty + gate_s_m.unsqueeze(1) * self.sty_mlp(sty_mod)
        
        return img, sty


class AdaptiveSingleStreamBlock(AdaptiveFluxBlock):
    """
    Adaptive Single-Stream Block with enhanced AdaLN.
    Processes unified features with self-attention.
    
    Improvements:
    - Enhanced AdaLN control for unified stream
    - Better style propagation through the network
    - Optional sparse activation for MoE
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_sparse=False):
        super().__init__(hidden_size, num_heads, mlp_ratio, use_sparse)
        
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.qk_norm = QKNorm(self.head_dim)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        
        # AdaLN with 6 components (scale, shift, gate for attention and MLP)
        self.adaLN = AdaLNModulation(hidden_size, components=6)
    
    def forward(self, x, vec, pe):
        """
        Forward pass with single-stream processing
        
        Args:
            x: Input features [B, L, D]
            vec: Conditioning vector [B, D]
            pe: Position embeddings
        
        Returns:
            Updated features
        """
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(vec)
        
        # Attention with AdaLN
        x_mod = self.apply_adaptive_norm(self.norm1(x), scale_a, shift_a)
        qkv = self.qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k)
        
        dtype = x.dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        
        # Apply RoPE if position embeddings provided
        if pe is not None:
            pe_reshaped = pe.transpose(1, 2)
            q, k = apply_rope(q, k, pe_reshaped)
            q, k = q.to(dtype), k.to(dtype)
        
        x_attn = self.proj(attention(q, k, v))
        x = x + gate_a.unsqueeze(1) * x_attn
        
        # MLP with AdaLN
        x_mod = self.apply_adaptive_norm(self.norm2(x), scale_m, shift_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(x_mod)
        
        return x


# For backward compatibility, create aliases
SwiftDoubleStreamBlock = AdaptiveDoubleStreamBlock
SwiftSingleStreamBlock = AdaptiveSingleStreamBlock


__all__ = [
    'AdaptiveFluxBlock',
    'AdaptiveDoubleStreamBlock',
    'AdaptiveSingleStreamBlock',
    'SwiftDoubleStreamBlock',
    'SwiftSingleStreamBlock',
    'QKNorm',
    'SwiGLU',
]
