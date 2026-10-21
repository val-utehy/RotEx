"""
Advanced attention mechanisms for better semantic and style alignment.
Includes Cross-Attention, AdaIN, and Depth-Controlled Attention.
Author: https://github.com/datnguyen-tien204
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional


class AdaptiveInstanceNorm2d(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.
    Aligns mean and variance of content features to match style features.
    """
    def __init__(self, num_features: int, style_dim: int):
        """
        Args:
            num_features: Number of feature channels
            style_dim: Dimension of style vector
        """
        super().__init__()
        self.num_features = num_features
        
        # Style modulation parameters
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        
        # Initialize to identity transformation
        nn.init.constant_(self.style_scale.weight, 0)
        nn.init.constant_(self.style_scale.bias, 1)
        nn.init.constant_(self.style_shift.weight, 0)
        nn.init.constant_(self.style_shift.bias, 0)
    
    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, C, H, W]
            style: Style vector [B, style_dim]
        Returns:
            normalized: Style-adapted features [B, C, H, W]
        """
        # Instance normalization
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized = (x - mean) / std
        
        # Style modulation
        scale = self.style_scale(style).unsqueeze(-1).unsqueeze(-1)
        shift = self.style_shift(style).unsqueeze(-1).unsqueeze(-1)
        
        return normalized * scale + shift


class AdaIN1d(nn.Module):
    """
    AdaIN for 1D features (sequence data).
    """
    def __init__(self, num_features: int, style_dim: int):
        super().__init__()
        self.num_features = num_features

        self.norm = nn.LayerNorm(num_features, elementwise_affine=False)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)

        nn.init.constant_(self.style_scale.weight, 0)
        nn.init.constant_(self.style_scale.bias, 1)
        nn.init.constant_(self.style_shift.weight, 0)
        nn.init.constant_(self.style_shift.bias, 0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, C]
            style: Style vector [B, style_dim] OR [B, L_s, style_dim]
        Returns:
            normalized: Style-adapted features [B, L, C]
        """
        if style.dim() == 3:
            style = style.mean(dim=1)

        normalized = self.norm(x)
        scale = self.style_scale(style).unsqueeze(1)
        shift = self.style_shift(style).unsqueeze(1)

        return normalized * scale + shift



class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for semantic alignment between content and style.
    """
    def __init__(self, query_dim: int, key_dim: int, num_heads: int = 8, 
                 dropout: float = 0.0):
        """
        Args:
            query_dim: Dimension of query features
            key_dim: Dimension of key/value features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        
        self.query = nn.Linear(query_dim, query_dim)
        self.key = nn.Linear(key_dim, query_dim)
        self.value = nn.Linear(key_dim, query_dim)
        
        self.proj = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: Query features [B, L_q, query_dim]
            key: Key features [B, L_k, key_dim]
            value: Value features [B, L_k, key_dim] (if None, uses key)
            mask: Attention mask [B, L_q, L_k] (optional)
        Returns:
            output: Attended features [B, L_q, query_dim]
        """
        if value is None:
            value = key
        
        B, L_q, _ = query.shape
        L_k = key.shape[1]
        
        # Linear projections
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b l (h d) -> b h l d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b h l d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b h l d', h=self.num_heads)
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        
        # Output projection
        out = self.proj(out)
        
        return out


class DepthControlledAttention(nn.Module):
    """
    Depth-controlled attention mechanism for structure-aware feature learning.
    Uses depth information to modulate attention patterns.
    """
    def __init__(self, feature_dim: int, depth_dim: int = 64, num_heads: int = 8):
        """
        Args:
            feature_dim: Dimension of input features
            depth_dim: Dimension of depth features
            num_heads: Number of attention heads
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # Process depth information
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, depth_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(depth_dim, depth_dim, kernel_size=3, padding=1)
        )
        
        # Depth-modulated attention
        self.qkv = nn.Linear(feature_dim, feature_dim * 3)
        self.depth_gate = nn.Linear(depth_dim, feature_dim)
        self.proj = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, C]
            depth: Depth map [B, 1, H, W]
        Returns:
            output: Depth-modulated features [B, L, C]
        """
        B, L, C = x.shape
        
        # Encode depth
        depth_feat = self.depth_encoder(depth)  # [B, depth_dim, H, W]
        H, W = depth_feat.shape[-2:]
        depth_feat = rearrange(depth_feat, 'b c h w -> b (h w) c')
        
        # Generate Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b l (three h d) -> three b h l d', 
                           three=3, h=self.num_heads)
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Modulate attention with depth
        if depth_feat.shape[1] == L:
            depth_gate = self.depth_gate(depth_feat)
            depth_gate = rearrange(depth_gate, 'b l (h d) -> b h l d', h=self.num_heads)
            depth_mask = torch.sigmoid(depth_gate.mean(dim=-1, keepdim=True))
            attn = attn * depth_mask
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)')
        out = self.proj(out)
        
        return out


class StyleAttentionBlock(nn.Module):
    """
    Combined attention block with AdaIN and Cross-Attention.
    """
    def __init__(self, feature_dim: int, style_dim: int, num_heads: int = 8):
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Cross-attention for style conditioning
        self.cross_attn = CrossAttention(feature_dim, style_dim, num_heads)
        
        # AdaIN for style transfer
        self.adain = AdaIN1d(feature_dim, style_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
    
    def forward(self, x: torch.Tensor, style: torch.Tensor, 
                style_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, C]
            style: Style vector [B, style_dim]
            style_features: Optional style feature sequence [B, L_s, style_dim]
        Returns:
            output: Style-conditioned features [B, L, C]
        """
        # Cross-attention with style features
        if style_features is not None:
            attn_out = self.cross_attn(self.norm1(x), style_features)
            x = x + attn_out
        
        # AdaIN style transfer
        x = self.adain(x, style)
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        
        return x


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention for capturing features at different resolutions.
    """
    def __init__(self, feature_dim: int, num_scales: int = 3, num_heads: int = 8):
        super().__init__()
        self.num_scales = num_scales
        
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
            for _ in range(num_scales)
        ])
        
        self.fusion = nn.Linear(feature_dim * num_scales, feature_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features [B, L, C]
        Returns:
            output: Multi-scale attended features [B, L, C]
        """
        B, L, C = x.shape
        
        outputs = []
        for i, attn in enumerate(self.attentions):
            # Downsample for different scales
            if i > 0:
                scale = 2 ** i
                # Simple downsampling by reshaping
                L_down = L // scale
                if L_down > 0:
                    x_down = x[:, ::scale, :]
                else:
                    x_down = x
                out, _ = attn(x_down, x_down, x_down)
                # Upsample back
                if out.shape[1] != L:
                    out = F.interpolate(out.transpose(1, 2), size=L, mode='linear', 
                                      align_corners=False).transpose(1, 2)
            else:
                out, _ = attn(x, x, x)
            
            outputs.append(out)
        
        # Concatenate and fuse
        multi_scale = torch.cat(outputs, dim=-1)
        fused = self.fusion(multi_scale)
        
        return fused
