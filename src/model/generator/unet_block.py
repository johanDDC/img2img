import torch
import torch.nn as nn
import abc

class UnetBlock(nn.Module, abc.ABC):
    def __init__(self, norm_shape, in_channels, out_channels, dropout_prob):
        super().__init__()
        self.ln1 = nn.LayerNorm(norm_shape)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.gelu = nn.GELU()

    @abc.abstractmethod
    def forward(self, x):
        pass


class DownsamplingUnetBlock(UnetBlock):
    def __init__(self, norm_shape, in_channels, out_channels, dropout_prob):
        super().__init__(norm_shape, in_channels, out_channels, dropout_prob)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.ln1(x)
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.conv2(x)
        x = self.gelu(x)
        return x, self.pool(x)


class UpsamplingUnetBlock(UnetBlock):
    def __init__(self, norm_shape, in_channels, out_channels, dropout_prob):
        super().__init__(norm_shape, in_channels, out_channels, dropout_prob)
