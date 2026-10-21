"""
Advanced encoder modules for enhanced feature extraction.
Supports DINOv2, EVA-CLIP, and other state-of-the-art vision encoders.
Author: https://github.com/datnguyen-tien204
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoImageProcessor, AutoModelForDepthEstimation
from typing import Optional, Literal
from flux.utils.logger_utils import ConsoleLogger

class DepthEstimator:
    def __init__(self, device):
        self.device = device
        ConsoleLogger.info("Loading Depth Anything V2 Small...")
        self.proc = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()

    @torch.no_grad()
    def predict(self, images, size=384):
        imgs_255 = ((images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)
        inputs = self.proc(images=imgs_255, return_tensors="pt", size={"height": 518, "width": 518}).to(self.device)
        depth = self.model(**inputs).predicted_depth
        depth = F.interpolate(depth.unsqueeze(1), size=(size, size), mode="bicubic")
        for i in range(depth.shape[0]):
            depth[i] = (depth[i] - depth[i].min()) / (depth[i].max() - depth[i].min() + 1e-6)
        return depth

class DINOv2Encoder(nn.Module):
    """
    DINOv2 encoder for enhanced feature extraction.
    Better semantic understanding and structural preservation.
    """
    def __init__(self, device: str, output_dim: int = 512, 
                 model_size: Literal["small", "base", "large"] = "base"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        
        # Select model based on size
        model_names = {
            "small": "facebook/dinov2-small",
            "base": "facebook/dinov2-base",
            "large": "facebook/dinov2-large"
        }
        model_name = model_names[model_size]
        
        print(f"Loading DINOv2 encoder: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # Get hidden size from model config
        hidden_size = self.model.config.hidden_size
        
        # Projection layer to match output_dim
        self.proj = nn.Linear(hidden_size, output_dim, bias=False).to(device)
        
    def forward(self, x: torch.Tensor, return_spatial: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]
            return_spatial: Whether to return all patch tokens [B, L, D]
        Returns:
            features: [B, output_dim] or [B, L, output_dim]
        """
        # Normalize to [0, 1] for DINOv2
        x_normalized = (x + 1.0) / 2.0
        
        # Resize to 224x224 (DINOv2 standard input size)
        x_resized = F.interpolate(x_normalized, size=(224, 224), 
                                 mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=x_resized, return_dict=True)
            if return_spatial:
                # Return all patch tokens (excluding CLS if possible, though Flux handle sequences well)
                features = outputs.last_hidden_state
            else:
                # Use CLS token (first token) as global feature
                features = outputs.last_hidden_state[:, 0]
            
        return self.proj(features)


class EVACLIPEncoder(nn.Module):
    """
    EVA-CLIP encoder for enhanced vision-language feature extraction.
    State-of-the-art performance on visual understanding tasks.
    """
    def __init__(self, device: str, output_dim: int = 512,
                 model_size: Literal["base", "large"] = "large"):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        
        # EVA-CLIP models
        model_names = {
            "base": "BAAI/EVA-CLIP-8B",
            "large": "BAAI/EVA-CLIP-18B"
        }
        
        # Note: Using a more accessible model for now
        # In production, use actual EVA-CLIP if available
        model_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        
        print(f"Loading EVA-CLIP style encoder: {model_name}")
        try:
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to openai/clip-vit-large-patch14")
            from transformers import CLIPVisionModel
            self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
        self.model.eval()
        
        # Get hidden size
        hidden_size = self.model.config.hidden_size
        
        # Projection layer
        self.proj = nn.Linear(hidden_size, output_dim, bias=False).to(device)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W] in range [-1, 1]
        Returns:
            features: [B, output_dim]
        """
        # Normalize to [0, 1]
        x_normalized = (x + 1.0) / 2.0
        
        # Resize to 224x224
        x_resized = F.interpolate(x_normalized, size=(224, 224),
                                 mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            outputs = self.model(pixel_values=x_resized, return_dict=True)
            features = outputs.pooler_output
            
        return self.proj(features)


class DepthAwareEncoder(nn.Module):
    """
    Depth-aware feature encoder that combines visual features with depth information.
    """
    def __init__(self, base_encoder: nn.Module, depth_channels: int = 1, 
                 output_dim: int = 512):
        super().__init__()
        self.base_encoder = base_encoder
        self.output_dim = output_dim
        
        # Depth processing branch
        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim + 64, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor, depth: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W]
            depth: Depth maps [B, 1, H, W] (optional)
        Returns:
            features: [B, output_dim]
        """
        # Get base visual features
        visual_features = self.base_encoder(x)
        
        if depth is None:
            return visual_features
            
        # Process depth
        depth_features = self.depth_conv(depth)
        depth_features = depth_features.view(depth_features.size(0), -1)
        
        # Concatenate and fuse
        combined = torch.cat([visual_features, depth_features], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features


class EncoderFactory:
    """
    Factory class for creating different types of encoders.
    """
    @staticmethod
    def create_encoder(
        encoder_type: Literal["siglip", "dinov2", "evaclip"],
        device: str,
        output_dim: int = 512,
        depth_aware: bool = False,
        **kwargs
    ) -> nn.Module:
        """
        Create an encoder of the specified type.
        
        Args:
            encoder_type: Type of encoder to create
            device: Device to load the model on
            output_dim: Output dimension of the encoder
            depth_aware: Whether to wrap encoder with depth-aware processing
            **kwargs: Additional arguments for specific encoders
        
        Returns:
            encoder: Configured encoder module
        """
        if encoder_type == "siglip":
            from transformers import SiglipVisionModel
            
            class SigLIPStyleEncoder(nn.Module):
                def __init__(self, device, output_dim=512):
                    super().__init__()
                    model_name = "google/siglip-so400m-patch14-384"
                    self.model = SiglipVisionModel.from_pretrained(model_name).to(device)
                    self.model.eval()
                    self.proj = nn.Linear(self.model.config.hidden_size, output_dim, bias=False).to(device)
                
                def forward(self, x):
                    x_resized = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
                    with torch.no_grad():
                        outputs = self.model(pixel_values=x_resized)
                        features = outputs.last_hidden_state.mean(dim=1)
                    return self.proj(features)
            
            encoder = SigLIPStyleEncoder(device, output_dim)
            
        elif encoder_type == "dinov2":
            model_size = kwargs.get("model_size", "base")
            encoder = DINOv2Encoder(device, output_dim, model_size)
            
        elif encoder_type == "evaclip":
            model_size = kwargs.get("model_size", "large")
            encoder = EVACLIPEncoder(device, output_dim, model_size)
            
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # Wrap with depth-aware processing if requested
        if depth_aware:
            encoder = DepthAwareEncoder(encoder, depth_channels=1, output_dim=output_dim)
        
        return encoder
