import os
import numpy as np
import torch
from torchvision.transforms import ToTensor, RandomHorizontalFlip, Normalize, Resize
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from utils import Color

class ColorizationDataset(Dataset):
    def __init__(self, data_path, transform=None, valid=False, scale_factor=256):
        self.data = ImageFolder(data_path, transform=transform)
        self.horizontal_flip = RandomHorizontalFlip(1)
        self.normalize = Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.valid = valid
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # try:
        #
        # except:
        #     return self[(item + 1) % len(self)]
        image = self.data[item][0]
        if self.valid:
            new_h = image.shape[1] // self.scale_factor * self.scale_factor
            new_w = image.shape[2] // self.scale_factor * self.scale_factor
            image = Resize((new_h, new_w))(image)
        image = image.numpy().transpose(1, 2, 0)
        lab_image = Color.rgb2lab(image)
        stats = torch.tensor((lab_image.min(), lab_image.max()))
        lab_image = (lab_image - stats[0].item()) / (stats[1].item() - stats[0].item() + 1e-5)
        lab_image = ToTensor()(lab_image)
        lab_image = self.normalize(lab_image)
        return lab_image[[0]], lab_image[[1, 2]], stats
