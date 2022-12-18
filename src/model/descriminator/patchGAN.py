import torch
import torch.nn as nn
import torch.nn.init as init


class PatchGAN(nn.Module):
    def __init__(self, n_layers, in_channels, start_num_filters=64):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, start_num_filters,
                          kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        cur_channels = start_num_filters * 2
        for i in range(1, n_layers - 1):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(start_num_filters, cur_channels,
                              kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(cur_channels),
                    nn.LeakyReLU(0.2, inplace=True))
            )
            start_num_filters = cur_channels
            cur_channels *= 2

        self.layers[-1] = nn.Sequential(
            nn.Conv2d(start_num_filters // 2, start_num_filters,
                      kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(start_num_filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layers = nn.Sequential(
            *self.layers,
            nn.Conv2d(start_num_filters, 1,
                      kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
        )
        self.apply(self.__init_weights)

    @staticmethod
    def __init_weights(layer):
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            init.normal_(layer.weight, 0, 0.02)

    def forward(self, generator_out):
        return self.layers(generator_out)
