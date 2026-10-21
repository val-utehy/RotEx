"""
State-of-the-art sampling techniques (2024-2025).
Includes Flow Matching with Optimal Transport and Consistency Models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


class FlowMatching:
    """
    Flow Matching with Optimal Transport for improved trajectory learning.
    Based on recent advances in continuous normalizing flows.
    """
    def __init__(self, sigma_min: float = 0.001):
        """
        Args:
            sigma_min: Minimum noise level for numerical stability
        """
        self.sigma_min = sigma_min
    
    def compute_optimal_transport_path(self, x0: torch.Tensor, x1: torch.Tensor, 
                                      t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute optimal transport path from noise (x0) to data (x1).
        
        Args:
            x0: Source distribution (noise) [B, ...]
            x1: Target distribution (data) [B, ...]
            t: Time steps [B] in range [0, 1]
        Returns:
            x_t: Interpolated samples
            v_t: Vector field (velocity)
        """
        t = t.view(-1, *([1] * (x0.dim() - 1)))
        
        # Optimal transport interpolation (linear for Gaussian)
        x_t = (1 - t) * x0 + t * x1
        
        # Velocity is the derivative: v = dx/dt = x1 - x0
        v_t = x1 - x0
        
        return x_t, v_t
    
    def compute_conditional_flow(self, x0: torch.Tensor, x1: torch.Tensor,
                                t: torch.Tensor, sigma: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute conditional flow with added noise for robustness.
        
        Args:
            x0: Noise [B, ...]
            x1: Data [B, ...]
            t: Time [B]
            sigma: Optional noise level (defaults to sigma_min * (1-t))
        Returns:
            x_t: Noisy interpolation
            v_t: Target velocity
        """
        if sigma is None:
            sigma = self.sigma_min * (1 - t.view(-1, *([1] * (x0.dim() - 1))))
        else:
            sigma = sigma * torch.ones_like(t).view(-1, *([1] * (x0.dim() - 1)))
        
        # Conditional path
        x_t, v_t = self.compute_optimal_transport_path(x0, x1, t)
        
        # Add Gaussian noise for conditional flow matching
        eps = torch.randn_like(x_t)
        x_t = x_t + sigma * eps
        
        return x_t, v_t
    
    def sample(self, model: Callable, x_init: torch.Tensor, 
              num_steps: int = 50, **model_kwargs) -> torch.Tensor:
        """
        Sample using Euler method for ODE solving.
        
        Args:
            model: Velocity prediction model
            x_init: Initial noise [B, ...]
            num_steps: Number of integration steps
            **model_kwargs: Additional arguments for model
        Returns:
            x_final: Generated samples
        """
        x = x_init
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_current = torch.full((x.shape[0],), step / num_steps, 
                                  device=x.device, dtype=x.dtype)
            
            # Predict velocity
            with torch.no_grad():
                v = model(x, timesteps=t_current, **model_kwargs)
            
            # Euler step
            x = x + v * dt
        
        return x
    
    def sample_heun(self, model: Callable, x_init: torch.Tensor,
                   num_steps: int = 50, **model_kwargs) -> torch.Tensor:
        """
        Higher-order sampling using Heun's method (2nd order).
        More accurate but requires 2x model evaluations.
        
        Args:
            model: Velocity prediction model
            x_init: Initial noise
            num_steps: Number of steps
            **model_kwargs: Model arguments
        Returns:
            x_final: Generated samples
        """
        x = x_init
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_current = torch.full((x.shape[0],), step / num_steps,
                                  device=x.device, dtype=x.dtype)
            
            # First stage
            with torch.no_grad():
                v1 = model(x, timesteps=t_current, **model_kwargs)
            
            x_temp = x + v1 * dt
            
            # Second stage
            t_next = torch.full((x.shape[0],), (step + 1) / num_steps,
                               device=x.device, dtype=x.dtype)
            
            with torch.no_grad():
                v2 = model(x_temp, timesteps=t_next, **model_kwargs)
            
            # Heun's method: average of two velocities
            x = x + (v1 + v2) * 0.5 * dt
        
        return x


class ConsistencyModel:
    """
    Consistency Models for fast, single-step generation.
    Based on "Consistency Models" (Song et al., 2023).
    """
    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0,
                 rho: float = 7.0):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Schedule parameter
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
    
    def get_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """
        Get noise schedule for consistency training.
        
        Args:
            num_steps: Number of discretization steps
            device: Device for tensors
        Returns:
            sigmas: Noise levels [num_steps + 1]
        """
        rho_inv = 1.0 / self.rho
        steps = torch.arange(num_steps + 1, device=device)
        sigmas = (self.sigma_max ** rho_inv + 
                 steps / num_steps * (self.sigma_min ** rho_inv - self.sigma_max ** rho_inv)) ** self.rho
        return sigmas
    
    def consistency_function(self, model: Callable, x: torch.Tensor,
                            sigma: torch.Tensor, **model_kwargs) -> torch.Tensor:
        """
        Apply consistency function: maps (x, sigma) to denoised output.
        
        Args:
            model: Denoising model
            x: Noisy input
            sigma: Noise level
            **model_kwargs: Model arguments
        Returns:
            denoised: Consistency function output
        """
        # Skip connection for sigma_min
        c_skip = self.sigma_min ** 2 / (sigma ** 2 + self.sigma_min ** 2)
        c_out = sigma * self.sigma_min / (sigma ** 2 + self.sigma_min ** 2).sqrt()
        
        # Predict denoised output
        with torch.no_grad():
            # Convert sigma to timestep for model
            t = sigma / self.sigma_max
            model_out = model(x, timesteps=t, **model_kwargs)
        
        # Apply consistency function
        denoised = c_skip * x + c_out * model_out
        
        return denoised
    
    def sample_onestep(self, model: Callable, shape: tuple,
                      device: torch.device, dtype: torch.dtype,
                      **model_kwargs) -> torch.Tensor:
        """
        Single-step generation from noise to data.
        
        Args:
            model: Consistency model
            shape: Output shape
            device: Device
            dtype: Data type
            **model_kwargs: Model arguments
        Returns:
            samples: Generated samples
        """
        # Sample from prior (high noise)
        x = torch.randn(*shape, device=device, dtype=dtype) * self.sigma_max
        sigma = torch.full((shape[0],), self.sigma_max, device=device, dtype=dtype)
        
        # One-step denoising
        samples = self.consistency_function(model, x, sigma, **model_kwargs)
        
        return samples
    
    def sample_multistep(self, model: Callable, shape: tuple,
                        num_steps: int, device: torch.device,
                        dtype: torch.dtype, **model_kwargs) -> torch.Tensor:
        """
        Multi-step generation for better quality.
        
        Args:
            model: Consistency model
            shape: Output shape
            num_steps: Number of steps
            device: Device
            dtype: Data type
            **model_kwargs: Model arguments
        Returns:
            samples: Generated samples
        """
        # Get noise schedule
        sigmas = self.get_schedule(num_steps, device)
        
        # Start from maximum noise
        x = torch.randn(*shape, device=device, dtype=dtype) * self.sigma_max
        
        # Iterative refinement
        for i in range(num_steps, 0, -1):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i - 1]
            
            # Denoise to current level
            x = self.consistency_function(
                model, x, 
                torch.full((shape[0],), sigma_current, device=device, dtype=dtype),
                **model_kwargs
            )
            
            # Add noise for next iteration (except last)
            if i > 1:
                noise = torch.randn_like(x)
                x = x + noise * (sigma_next ** 2 - self.sigma_min ** 2).sqrt()
        
        return x


class ClassifierFreeGuidance:
    """
    Enhanced Classifier-Free Guidance for better conditioning.
    """
    def __init__(self, guidance_scale: float = 7.5, min_scale: float = 1.0):
        """
        Args:
            guidance_scale: Guidance strength
            min_scale: Minimum guidance scale
        """
        self.guidance_scale = guidance_scale
        self.min_scale = min_scale
    
    def apply_guidance(self, cond_out: torch.Tensor, uncond_out: torch.Tensor,
                      guidance_scale: Optional[float] = None) -> torch.Tensor:
        """
        Apply classifier-free guidance to model outputs.
        
        Args:
            cond_out: Conditional model output
            uncond_out: Unconditional model output
            guidance_scale: Optional override for guidance scale
        Returns:
            guided_out: Guided output
        """
        scale = guidance_scale if guidance_scale is not None else self.guidance_scale
        scale = max(scale, self.min_scale)
        
        # CFG formula: out = uncond + scale * (cond - uncond)
        guided_out = uncond_out + scale * (cond_out - uncond_out)
        
        return guided_out
    
    def apply_dynamic_guidance(self, cond_out: torch.Tensor, uncond_out: torch.Tensor,
                              t: torch.Tensor, min_t: float = 0.0, 
                              max_t: float = 1.0) -> torch.Tensor:
        """
        Apply time-dependent guidance that decreases over time.
        
        Args:
            cond_out: Conditional output
            uncond_out: Unconditional output
            t: Time steps [B]
            min_t: Time to start decreasing guidance
            max_t: Time to reach minimum guidance
        Returns:
            guided_out: Dynamically guided output
        """
        # Linearly decrease guidance scale from max to min over time
        t_normalized = (t - min_t) / (max_t - min_t)
        t_normalized = t_normalized.clamp(0, 1)
        
        # Scale goes from guidance_scale to min_scale
        scale = self.guidance_scale + (self.min_scale - self.guidance_scale) * t_normalized
        scale = scale.view(-1, *([1] * (cond_out.dim() - 1)))
        
        guided_out = uncond_out + scale * (cond_out - uncond_out)
        
        return guided_out


class AdaptiveSampler:
    """
    Adaptive sampling that combines multiple techniques for optimal quality/speed.
    """
    def __init__(self, use_flow_matching: bool = True,
                 use_consistency: bool = False,
                 use_cfg: bool = True):
        self.use_flow_matching = use_flow_matching
        self.use_consistency = use_consistency
        self.use_cfg = use_cfg
        
        if use_flow_matching:
            self.flow_matching = FlowMatching()
        if use_consistency:
            self.consistency_model = ConsistencyModel()
        if use_cfg:
            self.cfg = ClassifierFreeGuidance()
    
    def sample(self, model: Callable, x_init: torch.Tensor,
              num_steps: int = 50, method: str = "heun",
              cfg_scale: Optional[float] = None,
              **model_kwargs) -> torch.Tensor:
        """
        Adaptive sampling with multiple techniques.
        
        Args:
            model: Model for sampling
            x_init: Initial noise
            num_steps: Number of steps
            method: Sampling method ("euler", "heun", "consistency")
            cfg_scale: Optional CFG scale
            **model_kwargs: Model arguments
        Returns:
            samples: Generated samples
        """
        if method == "consistency" and self.use_consistency:
            return self.consistency_model.sample_multistep(
                model, x_init.shape, num_steps,
                x_init.device, x_init.dtype, **model_kwargs
            )
        elif method == "heun" and self.use_flow_matching:
            return self.flow_matching.sample_heun(
                model, x_init, num_steps, **model_kwargs
            )
        else:  # euler
            return self.flow_matching.sample(
                model, x_init, num_steps, **model_kwargs
            )
