import torch
from torch import nn
from einops import rearrange, repeat
from flux.modules.layers import EmbedND, timestep_embedding, MLPEmbedder
from flux.utils.math import attention, apply_rope
from flux.modules.attention import AdaIN1d, DepthControlledAttention
from flux.modules.moe import MoEStyleAdapter, MoEFeatureProcessor
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, Any

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(hidden_features * 2 / 3)
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, out_features)
        self.silu = nn.SiLU()

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.silu(x1) * x2
        return self.w3(hidden)

class AdaLNModulation(nn.Module):
    def __init__(self, hidden_size, components=6):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(hidden_size, components * hidden_size, bias=True)
        self.components = components
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, vec):
        chunk = self.linear(self.silu(vec))
        return chunk.chunk(self.components, dim=1)

class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)
        self.key_norm = nn.LayerNorm(dim, elementwise_affine=True, eps=1e-6)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.query_norm(q), self.key_norm(k)

class ChronFluxDoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_depth_attn=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_depth_attn = use_depth_attn
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.img_proj = nn.Linear(hidden_size, hidden_size)
        self.sty_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sty_qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.sty_proj = nn.Linear(hidden_size, hidden_size)
        self.img_qk_norm = QKNorm(self.head_dim)
        self.sty_qk_norm = QKNorm(self.head_dim)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.sty_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.sty_mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
        self.adaLN = AdaLNModulation(hidden_size, components=12)
        if use_depth_attn:
            self.depth_attn = DepthControlledAttention(hidden_size, num_heads=num_heads)
        self.use_adain = False

    def set_adain(self, use_adain, style_dim):
        self.use_adain = use_adain
        if use_adain:
            self.adain = AdaIN1d(self.hidden_size, style_dim)

    def forward(self, img, sty, vec, pe, depth_map=None, style_vec=None):
        if self.use_adain and style_vec is not None:
            adain_style = style_vec
            if style_vec.dim() == 3:
                adain_style = style_vec.mean(dim=1)
            img = self.adain(img, adain_style)

        params = self.adaLN(vec)
        (shift_i_a, scale_i_a, gate_i_a, shift_i_m, scale_i_m, gate_i_m,
         shift_s_a, scale_s_a, gate_s_a, shift_s_m, scale_s_m, gate_s_m) = params

        img_mod = self.img_norm1(img) * (1 + scale_i_a.unsqueeze(1)) + shift_i_a.unsqueeze(1)
        sty_mod = self.sty_norm1(sty) * (1 + scale_s_a.unsqueeze(1)) + shift_s_a.unsqueeze(1)
        qkv_i = self.img_qkv(img_mod)
        qkv_s = self.sty_qkv(sty_mod)
        q_i, k_i, v_i = rearrange(qkv_i, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q_s, k_s, v_s = rearrange(qkv_s, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q_i, k_i = self.img_qk_norm(q_i, k_i)
        q_s, k_s = self.sty_qk_norm(q_s, k_s)
        dtype = img.dtype
        q_i, k_i, v_i = q_i.to(dtype), k_i.to(dtype), v_i.to(dtype)
        q_s, k_s, v_s = q_s.to(dtype), k_s.to(dtype), v_s.to(dtype)
        L_s_tokens = sty.shape[1]
        pe_s = pe[:, :, :L_s_tokens, ...]
        pe_i = pe[:, :, L_s_tokens:, ...]
        q_i, k_i = apply_rope(q_i, k_i, pe_i)
        q_s, k_s = apply_rope(q_s, k_s, pe_s)
        q_i, k_i = q_i.to(dtype), k_i.to(dtype)
        q_s, k_s = q_s.to(dtype), k_s.to(dtype)
        q_i = rearrange(q_i, "B H L D -> B L H D")
        k_i = rearrange(k_i, "B H L D -> B L H D")
        v_i = rearrange(v_i, "B H L D -> B L H D")
        q_s = rearrange(q_s, "B H L D -> B L H D")
        k_s = rearrange(k_s, "B H L D -> B L H D")
        v_s = rearrange(v_s, "B H L D -> B L H D")
        full_k = torch.cat([k_s, k_i], dim=1)
        full_v = torch.cat([v_s, v_i], dim=1)
        img_attn_out = self.img_proj(attention(q_i, full_k, full_v))
        sty_attn_out = self.sty_proj(attention(q_s, full_k, full_v))
        img = img + gate_i_a.unsqueeze(1) * img_attn_out
        sty = sty + gate_s_a.unsqueeze(1) * sty_attn_out
        if self.use_depth_attn and depth_map is not None:
            img = img + self.depth_attn(img, depth_map)
        img_mod = self.img_norm2(img) * (1 + scale_i_m.unsqueeze(1)) + shift_i_m.unsqueeze(1)
        img = img + gate_i_m.unsqueeze(1) * self.img_mlp(img_mod)
        sty_mod = self.sty_norm2(sty) * (1 + scale_s_m.unsqueeze(1)) + shift_s_m.unsqueeze(1)
        sty = sty + gate_s_m.unsqueeze(1) * self.sty_mlp(sty_mod)
        return img, sty

class ChronFluxSingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_moe=False, moe_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_moe = use_moe
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.qk_norm = QKNorm(self.head_dim)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_moe:
            params = moe_params or {}
            num_experts = params.get('num_experts', 3) # Default safe value
            # Extract only valid args for MoEFeatureProcessor
            valid_keys = ['custom_centers', 'custom_widths']
            kwargs = {k: params[k] for k in valid_keys if k in params}

            self.moe_processor = MoEFeatureProcessor(hidden_size, num_experts=num_experts, **kwargs)
            self.mlp = None
        else:
            self.mlp = SwiGLU(hidden_size, int(hidden_size * mlp_ratio))
            self.moe_processor = None
        self.adaLN = AdaLNModulation(hidden_size, components=6)

    def forward(self, x, vec, pe, phi=None):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(vec)
        x_mod = self.norm1(x) * (1 + scale_a.unsqueeze(1)) + shift_a.unsqueeze(1)
        qkv = self.qkv(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.qk_norm(q, k)
        dtype = x.dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
        q, k = apply_rope(q, k, pe)
        q, k = q.to(dtype), k.to(dtype)
        q = rearrange(q, "B H L D -> B L H D")
        k = rearrange(k, "B H L D -> B L H D")
        v = rearrange(v, "B H L D -> B L H D")
        x_attn = self.proj(attention(q, k, v))
        x = x + gate_a.unsqueeze(1) * x_attn
        x_mod = self.norm2(x) * (1 + scale_m.unsqueeze(1)) + shift_m.unsqueeze(1)

        aux_loss = 0.0
        if self.use_moe and phi is not None:
            moe_out, aux = self.moe_processor(x_mod, phi, return_aux=True)
            x = x + gate_m.unsqueeze(1) * moe_out
            if aux is not None:
                aux_loss = aux['load_balance_loss']
        else:
            x = x + gate_m.unsqueeze(1) * self.mlp(x_mod)

        return x, aux_loss

class ChronFlux(nn.Module):
    def __init__(self, in_channels=16, hidden_size=1024, num_double_layers=8,
                 num_single_layers=16, num_heads=16, style_dim=512,
                 use_text=False, text_dim=1152,
                 use_moe=True, use_depth_attn=True, use_adain=False,
                 moe_params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.in_channels = in_channels
        self.gradient_checkpointing = False
        self.use_text = use_text
        self.style_dim = style_dim
        self.use_moe = use_moe
        self.use_depth_attn = use_depth_attn
        self.use_adain = use_adain
        self.img_in = nn.Linear(16 * 3, hidden_size)
        self.depth_proj = nn.Conv2d(1, 16, kernel_size=1)
        self.style_in = nn.Linear(style_dim, hidden_size)
        self.default_style = nn.Parameter(torch.randn(1, style_dim))
        self.time_mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        if self.use_text:
            self.txt_mlp = nn.Sequential(
                nn.Linear(text_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size)
            )
            self.style_adapter = nn.Sequential(
                nn.Linear(text_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, style_dim)
            )
            nn.init.zeros_(self.style_adapter[-1].weight)
            nn.init.zeros_(self.style_adapter[-1].bias)
        self.phi_mlp = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Pass MoE params
        if self.use_moe:
            params = moe_params or {}
            num_experts = params.get('num_experts', 3)
            valid_keys = ['custom_centers', 'custom_widths']
            kwargs = {k: params[k] for k in valid_keys if k in params}
            self.moe_style_adapter = MoEStyleAdapter(style_dim, hidden_dim=style_dim * 2, num_experts=num_experts, **kwargs)

        self.vector_in = MLPEmbedder(hidden_size, hidden_size)
        self.pe_embedder = EmbedND(dim=hidden_size // num_heads, theta=10000, axes_dim=[16, 24, 24])
        self.double_blocks = nn.ModuleList()
        for _ in range(num_double_layers):
            block = ChronFluxDoubleStreamBlock(hidden_size, num_heads, use_depth_attn=use_depth_attn)
            block.set_adain(use_adain, style_dim)
            self.double_blocks.append(block)
        self.single_blocks = nn.ModuleList([
            ChronFluxSingleStreamBlock(
                hidden_size, num_heads,
                use_moe=(use_moe and i >= num_single_layers // 2),
                moe_params=moe_params
            )
            for i in range(num_single_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_layer = nn.Linear(hidden_size, in_channels)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, x_t, content_latent, depth_map, style_vec, timesteps, phi, txt_emb=None):
        B, C, H, W = x_t.shape
        moe_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)

        content_normalized = content_latent
        depth_small = torch.nn.functional.interpolate(depth_map, size=(H, W), mode="nearest")
        depth_emb = self.depth_proj(depth_small)
        img_cat = torch.cat([x_t, content_normalized, depth_emb], dim=1)
        img_flat = rearrange(img_cat, 'b c h w -> b (h w) c')
        img = self.img_in(img_flat)

        if style_vec is None:
            style_vec = self.default_style.expand(B, -1).to(x_t.dtype)
            if self.use_text and txt_emb is not None:
                style_delta = self.style_adapter(txt_emb)
                style_vec = style_vec + style_delta

        if self.use_moe:
            style_vec, aux = self.moe_style_adapter(style_vec, phi, return_aux=True)
            if aux is not None:
                moe_loss += aux['load_balance_loss']

        style = self.style_in(style_vec)
        if style.dim() == 2:
            style = style.unsqueeze(1)
        L_s = style.shape[1]

        t_emb = timestep_embedding(timesteps, 256).to(x_t.device).to(x_t.dtype)
        t_vec = self.time_mlp(t_emb)
        phi_norm = (phi % (2 * torch.pi)) / (2 * torch.pi)
        phi_input = torch.cat([
            torch.sin(phi.unsqueeze(1)),
            torch.cos(phi.unsqueeze(1)),
            torch.sin(2 * phi.unsqueeze(1)),
            torch.cos(2 * phi.unsqueeze(1)),
            phi_norm.unsqueeze(1)
        ], dim=1)
        phi_vec = self.phi_mlp(phi_input)
        vec_input = t_vec + phi_vec
        if self.use_text and txt_emb is not None:
            txt_vec = self.txt_mlp(txt_emb)
            vec_input = vec_input + txt_vec
        vec = self.vector_in(vec_input)

        img_ids = torch.zeros(H, W, 3).to(x_t.device)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(H, device=x_t.device)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(W, device=x_t.device)[None, :]
        if self.training:
            img_ids += torch.randn_like(img_ids) * 0.5
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=B)
        pe = self.pe_embedder(img_ids)
        dtype = pe.dtype
        device = pe.device
        pe_dim = pe.shape[-3]
        style_pe = torch.zeros(B, 1, L_s, pe_dim, 2, 2, device=device, dtype=dtype)
        full_pe = torch.cat([style_pe, pe], dim=2)

        for block in self.double_blocks:
            if self.gradient_checkpointing and self.training:
                img, style = checkpoint.checkpoint(block, img, style, vec, full_pe, depth_map, style_vec, use_reentrant=False)
            else:
                img, style = block(img, style, vec, full_pe, depth_map, style_vec=style_vec)

        x = torch.cat([style, img], dim=1)

        for block in self.single_blocks:
            if self.gradient_checkpointing and self.training:
                x, block_loss = checkpoint.checkpoint(block, x, vec, full_pe, phi, use_reentrant=False)
            else:
                x, block_loss = block(x, vec, full_pe, phi)
            moe_loss += block_loss

        x = x[:, style.shape[1]:, :]
        x = self.final_norm(x)
        x = self.final_layer(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, moe_loss