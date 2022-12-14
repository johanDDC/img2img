import torch
import torch.nn as nn

from src.model.generator.unet_block import DownsamplingUnetBlock, UpsamplingUnetBlock



class Unet(nn.Module):
    def __init__(self, img_shape, n_layers, in_channels, out_channels,
                 dropout_prob=0.3):
        super().__init__()
        self.n_layers = n_layers
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        h, w = img_shape
        n_channels = in_channels
        for layer in range(self.n_layers):
            self.down_blocks.append(
                DownsamplingUnetBlock((n_channels, w, h), n_channels, out_channels, dropout_prob)
            )
            self.up_blocks.append(
                UpsamplingUnetBlock((out_channels * 2, w, h), out_channels * 2, n_channels, dropout_prob)
            )
            n_channels = out_channels
            out_channels *= 2
            h, w = h // 2, w // 2

        self.bottleneck = DownsamplingUnetBlock((n_channels, w, h), n_channels, out_channels, dropout_prob)

    def forward(self, img):
        residuals = []
        output = img
        for layer in range(self.n_layers):
            residual, output = self.down_blocks[layer](output)
            residuals.append(residual)
        output, _ = self.bottleneck(output)
        for layer in range(self.n_layers):
            pass






