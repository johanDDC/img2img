import torch
import torch.nn as nn
import abc

class DownsamplingUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4),
                              stride=2, padding=1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.layer = nn.Sequential(
            self.relu,
            self.conv,
            self.norm
        )

    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self, n_layers, input_channels, inner_channels):
        super().__init__()
        self.n_layers = n_layers
        curr_channels = 64
        self.layers = nn.ModuleList()
        while curr_channels <= inner_channels:
            self.layers.append(
                DownsamplingUnetBlock(input_channels, curr_channels)
            )
            input_channels = curr_channels
            curr_channels *= 2
        for i in range(n_layers - len(self.layers) - 1):
            self.layers.append(
                DownsamplingUnetBlock(inner_channels, inner_channels)
            )
        self.inner_most = DownsamplingUnetBlock(inner_channels, inner_channels)
        self.inner_most.norm = nn.Identity()


    def forward(self, img):
        residuals = []
        x = img
        for i in range(self.n_layers):
            x = self.layers[i](x)
            residuals.append(x)
        x = self.inner_most(x)
        return x
