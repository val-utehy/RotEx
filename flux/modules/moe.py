"""
Mixture-of-Experts (MoE) module for time-based style control.
Implements sparse expert selection based on time intervals.
Author: https://github.com/datnguyen-tien204
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TimeBasedGatingNetwork(nn.Module):
    def __init__(self, num_experts: int, input_dim: int = 2, temperature: float = 0.2,
                 custom_centers: Optional[List[float]] = None,
                 custom_widths: Optional[List[float]] = None):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature

        if custom_centers is not None:
            centers_rad = [c * (2 * math.pi / 24.0) for c in custom_centers]
            self.register_buffer('expert_centers', torch.tensor(centers_rad))
        else:
            centers = [i * (2 * math.pi / num_experts) for i in range(num_experts)]
            self.register_buffer('expert_centers', torch.tensor(centers))

        if custom_widths is not None:
            widths_rad = [(w * (2 * math.pi / 24.0)) / 2.0 for w in custom_widths]
            init_log_widths = torch.log(torch.tensor(widths_rad) + 1e-5)
            self.expert_log_widths = nn.Parameter(init_log_widths)
        else:
            self.expert_log_widths = nn.Parameter(torch.zeros(num_experts))

        self.min_width = 0.1

    def forward(self, phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phi = phi.view(-1, 1)
        centers = self.expert_centers.view(1, -1)

        diff = torch.remainder(phi - centers + math.pi, 2 * math.pi) - math.pi

        widths = F.softplus(self.expert_log_widths).view(1, -1) + self.min_width

        logits = -diff ** 2 / (2 * widths ** 2)
        weights = F.softmax(logits / self.temperature, dim=-1)

        return weights, logits

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts, k=2, temperature=0.2, noise_std=0.1,
                 custom_centers=None, custom_widths=None):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std

        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])

        self.gate = TimeBasedGatingNetwork(
            num_experts, input_dim=2, temperature=temperature,
            custom_centers=custom_centers, custom_widths=custom_widths
        )
        self.register_buffer('load_balance_loss_coef', torch.tensor(0.01))

    def _add_noise(self, logits):
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            return logits + noise
        return logits

    def _compute_load_balance_loss(self, weights):
        mean_weights = weights.mean(dim=0)
        target = torch.ones_like(mean_weights) / self.num_experts
        loss = F.mse_loss(mean_weights, target)
        return loss * self.load_balance_loss_coef

    def forward(self, x, phi, return_aux=False):
        is_sequence = x.dim() == 3
        if is_sequence:
            B, L, C = x.shape
            x = x.reshape(B * L, C)
            phi = phi.unsqueeze(1).expand(B, L).reshape(B * L)

        weights, gate_logits = self.gate(phi)

        if self.training:
            gate_logits = self._add_noise(gate_logits)
            weights = F.softmax(gate_logits / self.gate.temperature, dim=-1)

        if self.k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(weights, self.k, dim=-1)
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            sparse_weights = torch.zeros_like(weights)
            sparse_weights.scatter_(1, top_k_indices, top_k_weights)
            weights = sparse_weights

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
        output = torch.einsum('be,beo->bo', weights, expert_outputs)

        if is_sequence:
            output = output.reshape(B, L, -1)
            weights = weights.reshape(B, L, -1).mean(dim=1)

        aux = None
        if return_aux:
            aux = {'load_balance_loss': self._compute_load_balance_loss(weights)}

        return output, aux

class MoEStyleAdapter(nn.Module):
    def __init__(self, style_dim, hidden_dim, num_experts, custom_centers=None, custom_widths=None):
        super().__init__()
        self.moe = MixtureOfExperts(
            style_dim, hidden_dim, style_dim, num_experts, k=2, temperature=0.2, noise_std=0.1,
            custom_centers=custom_centers, custom_widths=custom_widths
        )
    def forward(self, style, phi, return_aux=False):
        adapted, aux = self.moe(style, phi, return_aux=return_aux)
        return adapted + style, aux

class MoEFeatureProcessor(nn.Module):
    def __init__(self, feature_dim, num_experts, expand_ratio=4, custom_centers=None, custom_widths=None):
        super().__init__()
        self.moe = MixtureOfExperts(
            feature_dim, feature_dim * expand_ratio, feature_dim, num_experts, k=2, temperature=0.2, noise_std=0.1,
            custom_centers=custom_centers, custom_widths=custom_widths
        )
        self.norm = nn.LayerNorm(feature_dim)
    def forward(self, x, phi, return_aux=False):
        normed = self.norm(x)
        processed, aux = self.moe(normed, phi, return_aux=return_aux)
        return x + processed, aux