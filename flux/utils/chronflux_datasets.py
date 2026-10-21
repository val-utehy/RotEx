"""
    Author: https://github.com/datnguyen-tien204
"""

import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DayNightDataset(Dataset):
    def __init__(self, day_dir, night_dir, size=384):
        self.day_files = [os.path.join(day_dir, f) for f in os.listdir(day_dir) if f.endswith(('.jpg', '.png'))]
        self.night_files = [os.path.join(night_dir, f) for f in os.listdir(night_dir) if f.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    def __len__(self): return len(self.day_files)

    def __getitem__(self, idx):
        day_img = Image.open(self.day_files[idx]).convert('RGB')
        night_img = Image.open(np.random.choice(self.night_files)).convert('RGB')
        return self.transform(day_img), self.transform(night_img)