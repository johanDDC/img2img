import torch
import torch.nn as nn


class UpsamplingUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_prob, inplace=True) if dropout_prob > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.layer(x)


class Decoder(nn.Module):
    def __init__(self, n_layers, inner_channels, out_channels):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        top_layers = nn.ModuleList()
        curr_channels = inner_channels
        self.layers.append(
            UpsamplingUnetBlock(inner_channels, inner_channels)
        )
        while inner_channels > out_channels:
            top_layers.append(
                UpsamplingUnetBlock(inner_channels * 2, inner_channels // 2, 0)
            )
            inner_channels //= 2
        rest_layers_cnt = n_layers - len(top_layers) - 1
        for i in range(rest_layers_cnt):
            layer = UpsamplingUnetBlock(2 * curr_channels, curr_channels)
            self.layers.append(layer)
        self.layers.extend(top_layers)

    def forward(self, encoder_out, residuals):
        input = encoder_out
        x = self.layers[0](input)
        for i in range(1, self.n_layers):
            input = torch.cat([residuals[-i], x], dim=1)
            x = self.layers[i](input)
        return torch.cat([residuals[0], x], dim=1)
