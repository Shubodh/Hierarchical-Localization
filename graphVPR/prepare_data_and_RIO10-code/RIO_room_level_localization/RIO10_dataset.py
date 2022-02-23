import os

import numpy as np
import cv2
from pathlib import Path
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torch import cuda

from utils import read_image

class RIO10Dataset(Dataset):
    def __init__(self, base_path, sample_path, room, sampling_freq, device):
        self.base_path = base_path
        self.sample_path = sample_path
        self.room = room
        self.sampling_freq = sampling_freq
        self.device = device

        if base_path == sample_path:
            full_path = os.path.join(base_path, room)
        else:
            full_path = os.path.join(base_path, f"scene{room[:2]}", f"seq{room[:2]}", f"seq{room[:2]}")

        full_path = Path(full_path)

        self.img_files = sorted(list(full_path.glob("*color.jpg")))
        self.img_files = self.img_files[::sampling_freq]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = read_image(self.img_files[index]).astype(np.float32)
        img = np.moveaxis(np.array(img), -1, 0)
        img = torch.from_numpy(img).float().to(device=self.device)
        return img

def get_RIO10_data(base_path, sample_path, room, sampling_freq, device, batch_size, shuffle=False, num_workers=4):
    dataset = RIO10Dataset(base_path, sample_path, room, sampling_freq, device)

    dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, len(dataset)






