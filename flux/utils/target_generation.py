"""
    Author: https://github.com/datnguyen-tien204
"""
import numpy as np
import torch
import math

PI = math.pi

def symmetric_brightness_curve(hour):
    return 0.035 + (1.0 - 0.035) * (0.5 - 0.5 * np.cos(2 * PI * (hour / 24.0)))


def rgb_curve_original(hour):
    t = (hour % 24.0) / 24.0
    sunset = np.exp(-((t - 0.75) / 0.08) ** 2)
    sunrise = np.exp(-((t - 0.25) / 0.08) ** 2)
    night = np.exp(-((t - 0.0) / 0.10) ** 2) + np.exp(-((t - 1.0) / 0.10) ** 2)
    noon = np.exp(-((t - 0.5) / 0.12) ** 2)
    return [1.0 + 0.4 * sunset + 0.22 * sunrise - 0.35 * night,
            1.0 - 0.1 * sunset - 0.06 * night,
            1.0 - 0.5 * sunset + 0.3 * night + 0.15 * noon]


def create_target_final_v2(day_img, style_img, depth_map, hour_val, adain_module, style_stats):
    device, dtype, B = day_img.device, day_img.dtype, day_img.shape[0]
    img_original = (day_img + 1.0) / 2.0
    target_list = []

    for i in range(B):
        h = hour_val[i].item()
        bright = symmetric_brightness_curve(h)
        rgb = rgb_curve_original(h)

        tint = torch.tensor(rgb, device=device, dtype=dtype).view(1, 3, 1, 1)
        tint = tint * (0.6 + 0.4 * style_stats["mean"][i].view(1, 3, 1, 1))

        img_i = img_original[i:i + 1]
        target_base = img_i * tint

        if bright < 0.8:
            adain_out = (adain_module(day_img[i:i + 1], style_img[i:i + 1]) + 1.0) / 2.0
            night_int = (1.0 - bright) ** 0.8 * 0.85

            d_i = depth_map[i:i + 1]
            d_norm = (d_i - d_i.min()) / (d_i.max() - d_i.min() + 1e-6)
            sky_mask = torch.sigmoid((0.15 - d_norm) * 10.0)

            blend = night_int * (0.7 + 0.3 * sky_mask)
            target_mixed = target_base * (1 - blend) + adain_out * blend
        else:
            target_mixed = target_base
            sky_mask = torch.zeros_like(target_mixed[:, :1])

        target_calc = target_mixed * bright
        if bright < 0.20:
            target_calc += img_i.mean(1, keepdim=True) * tint * 0.03
            if bright < 0.8: target_calc *= (1.0 - sky_mask * 0.5)

        noon_w = torch.tensor(np.exp(-(abs(h - 12.0) ** 2) / 8.0), device=device, dtype=dtype)
        target_list.append(target_calc * (1 - noon_w) + img_i * noon_w)

    return (torch.cat(target_list, dim=0) * 2.0 - 1.0).clamp(-1, 1)