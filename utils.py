import random
import torch

import numpy as np
from numpy import linalg


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class Color:
    xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
                             [0.212671, 0.715160, 0.072169],
                             [0.019334, 0.119193, 0.950227]])

    rgb_from_xyz = linalg.inv(xyz_from_rgb)

    @staticmethod
    def lab2rgb(lab):
        return Color.__xyz2rgb(Color.__lab2xyz(lab))

    @staticmethod
    def rgb2lab(rgb):
        return Color.__xyz2lab(Color.__rgb2xyz(rgb))

    @staticmethod
    def __xyz2rgb(xyz):
        arr = xyz @ Color.rgb_from_xyz.T
        mask = arr > 0.0031308
        arr[mask] = 1.055 * np.power(arr[mask], 1 / 2.4) - 0.055
        arr[~mask] *= 12.92
        return np.clip(arr, 0, 1)

    @staticmethod
    def __lab2xyz(lab):
        arr = lab.numpy()

        L, a, b = arr[..., 0], arr[..., 1], arr[..., 2]
        y = (L + 16.) / 116.
        x = (a / 500.) + y
        z = y - (b / 200.)

        if np.any(z < 0):
            invalid = np.nonzero(z < 0)
            z[invalid] = 0

        out = np.stack([x, y, z], axis=-1)

        mask = out > 0.2068966
        out[mask] = np.power(out[mask], 3.)
        out[~mask] = (out[~mask] - 16.0 / 116.) / 7.787

        xyz_ref_white = np.asarray((0.95047, 1., 1.08883), dtype=np.float32)
        out *= xyz_ref_white
        return out

    @staticmethod
    def __xyz2lab(xyz):
        arr = xyz

        xyz_ref_white = np.asarray((0.95047, 1., 1.08883), dtype=np.float32)
        arr = arr / xyz_ref_white

        mask = arr > 0.008856
        arr[mask] = np.cbrt(arr[mask])
        arr[~mask] = 7.787 * arr[~mask] + 16. / 116.

        x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]

        # Vector scaling
        L = (116. * y) - 16.
        a = 500.0 * (x - y)
        b = 200.0 * (y - z)

        return np.concatenate([x[..., np.newaxis] for x in [L, a, b]], axis=-1)

    @staticmethod
    def __rgb2xyz(rgb):
        arr = rgb.copy()
        mask = arr > 0.04045
        arr[mask] = np.power((arr[mask] + 0.055) / 1.055, 2.4)
        arr[~mask] /= 12.92
        return arr @ Color.xyz_from_rgb.T.astype(arr.dtype)


def freeze_weights(model, freeze=True):
    for param in model.parameters():
        param.requires_grad_(not freeze)


class Dict(dict):
    def __init__(self, dct=None):
        super().__init__()
        dct = dict() if not dct else dct
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = Dict(value)
            self[key] = value

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
