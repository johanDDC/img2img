import torch
import torch.nn as nn
import abc

class DownsamplingUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True, relu=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True) if relu else nn.Identity(),
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4),
                      stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)

class Encoder(nn.Module):
    def __init__(self, n_layers, input_channels, inner_channels, start_num_filters):
        super().__init__()
        self.n_layers = n_layers
        curr_channels = start_num_filters
        self.layers = nn.ModuleList()
        self.layers.append(
            DownsamplingUnetBlock(input_channels, curr_channels, norm=False, relu=False)
        )
        input_channels = curr_channels
        curr_channels *= 2

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
        self.inner_most = DownsamplingUnetBlock(inner_channels, inner_channels, norm=False)


    def forward(self, img):
        residuals = []
        x = img
        for i in range(self.n_layers - 1):
            x = self.layers[i](x)
            residuals.append(x)
        x = self.inner_most(x)
        return x, residuals
