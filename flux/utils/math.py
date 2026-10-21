"""
Author: https://github.com/datnguyen-tien204
"""

import torch
from einops import rearrange
from torch import Tensor
from flash_attn import flash_attn_func


def attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    # Ensure tensors are in bf16 or fp16 for FlashAttention =====
    original_dtype = q.dtype
    if q.dtype not in [torch.float16, torch.bfloat16]:
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
    # ======================================================================

    x = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
    x = rearrange(x, "B L H D -> B L (H D)")

    # Convert back to original dtype if needed =====
    if original_dtype not in [torch.float16, torch.bfloat16]:
        x = x.to(original_dtype)
    # ==========================================================

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)

    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]

    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)