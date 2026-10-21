"""
Perceptual and adversarial loss modules for enhanced training.
Includes LPIPS, VGG perceptual loss, and PatchGAN discriminator.
Author: https://github.com/datnguyen-tien204
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms, models
from typing import List, Optional
import lpips


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
    

class DayPreservationLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred_img, day_img, hour_val):
        if isinstance(hour_val, torch.Tensor):
            weights = []
            for h in hour_val:
                h = h.item() % 24.0
                if 11.0 <= h <= 13.0: w = 1.0
                elif 10.0 <= h < 11.0: w = (h - 10.0)
                elif 13.0 < h <= 14.0: w = (14.0 - h)
                else: w = 0.0
                weights.append(w)
            weight_tensor = torch.tensor(weights, device=pred_img.device, dtype=pred_img.dtype).view(-1, 1, 1, 1)
            if weight_tensor.sum() > 0:
                return (torch.abs(pred_img - day_img) * weight_tensor).mean()
        return torch.tensor(0.0, device=pred_img.device)

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters(): param.requires_grad = False
        self.slice1 = nn.Sequential(*list(vgg[:4])).to(device)
        self.slice2 = nn.Sequential(*list(vgg[4:9])).to(device)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, y):
        x_norm = self.normalize((x + 1.0) / 2.0)
        y_norm = self.normalize((y + 1.0) / 2.0)
        x_f1 = self.slice1(x_norm)
        y_f1 = self.slice1(y_norm)
        x_f2 = self.slice2(x_f1)
        y_f2 = self.slice2(y_f1)
        return (F.l1_loss(x_f1, y_f1) * 0.5 + F.l1_loss(x_f2, y_f2) * 0.5) * 0.01

class ColorMatchingLoss(nn.Module):
    def __init__(self, device, use_perceptual=True):
        super().__init__()
        self.use_perceptual = use_perceptual
        if use_perceptual: self.perceptual_loss = PerceptualLoss(device)
    def forward(self, pred, target):
        pred_01, target_01 = (pred + 1) / 2, (target + 1) / 2
        mean_loss = F.mse_loss(pred_01.mean([2,3]), target_01.mean([2,3]))
        std_loss = F.mse_loss(pred_01.std([2,3]), target_01.std([2,3]))

        eps = 1e-3
        p_c, t_c = pred_01.clamp(eps, 1), target_01.clamp(eps, 1)
        def ratios(x): return (torch.log(x[:,0:1])-torch.log(x[:,1:2])).clamp(-3,3), (torch.log(x[:,0:1])-torch.log(x[:,2:3])).clamp(-3,3), (torch.log(x[:,1:2])-torch.log(x[:,2:3])).clamp(-3,3)
        prg, prb, pgb = ratios(p_c)
        trg, trb, tgb = ratios(t_c)
        ratio_loss = (F.mse_loss(prg, trg) + F.mse_loss(prb, trb) + F.mse_loss(pgb, tgb)) / 3

        loss = mean_loss * 1.0 + std_loss * 0.5 + ratio_loss * 0.3
        if self.use_perceptual: loss += self.perceptual_loss(pred, target)
        return loss

class ContinuityLoss(nn.Module):
    def __init__(self, max_rate=0.15):
        super().__init__()
        self.max_rate = max_rate

    def forward(self, outputs, phis):
        # outputs: [N, 3, H, W] - N can be batch size OR temporal buffer size
        if outputs.shape[0] < 2: return torch.tensor(0.0, device=outputs.device)

        # Sort by phi to ensure temporal order
        idx = torch.argsort(phis)
        out, phi = outputs[idx], phis[idx]

        # Calculate Brightness
        b = 0.299*out[:,0]+0.587*out[:,1]+0.114*out[:,2]

        # Calculate diffs
        diff_phi = (phi[1:] - phi[:-1])

        # Filter out cases where phi is too close (avoid division by zero)
        mask = diff_phi > 1e-4
        if not mask.any(): return torch.tensor(0.0, device=outputs.device)

        # Rate of change: |dB| / dPhi
        dB = torch.abs(b[1:] - b[:-1])
        dB = dB[mask]                         # [K, H, W]
        dphi = diff_phi[mask].view(-1,1,1)    # [K,1,1]

        rate = dB / dphi
        return torch.mean(F.relu(rate - self.max_rate).clamp(0, 0.5)**2)

class ColorContinuityLoss(nn.Module):
    def __init__(self, max_change=0.10):
        super().__init__()
        self.max_change = max_change

    def forward(self, outputs, phis):
        if outputs.shape[0] < 2: return torch.tensor(0.0, device=outputs.device)

        idx = torch.argsort(phis)
        out, phi = outputs[idx], phis[idx]

        diff_phi = (phi[1:] - phi[:-1])
        mask = diff_phi > 1e-4
        if not mask.any(): return torch.tensor(0.0, device=outputs.device)

        # Rate of change: |dColor| / dPhi
        rate = torch.abs(out[1:] - out[:-1])[mask] / diff_phi[mask].view(-1,1,1,1)

        loss = 0
        for c in range(3):
            loss += torch.mean(F.relu(rate[:,c] - self.max_change).clamp(0, 0.3)**2)
        return loss / 3