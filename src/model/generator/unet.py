import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.generator.decoder import Decoder
from src.model.generator.encoder import Encoder


class Unet(nn.Module):
    def __init__(self, n_layers, in_channels, inner_channels, out_channels, start_num_filters):
        super().__init__()
        self.n_layers = n_layers
        self.encoder = Encoder(n_layers, in_channels, inner_channels, start_num_filters)
        self.decoder = Decoder(n_layers - 1, inner_channels, start_num_filters)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2 * start_num_filters, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(layer.weight, 0, 0.02)
            init.constant_(layer.bias, 0)

    def forward(self, img):
        encoder_out, residuals = self.encoder(img)
        result = self.decoder(encoder_out, residuals)
        return self.head(result)
