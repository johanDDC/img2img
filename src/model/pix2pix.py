import torch
import torch.nn as nn
import torch.nn.init as init

from src.model.descriminator.patchGAN import PatchGAN
from src.model.generator.unet import Unet
from src.model.loss import Loss


class Pix2Pix(nn.Module):
    def __init__(self, generator_n_layers, descriminator_n_layers,
                 in_channels, inner_channels, out_channels, start_num_filters,
                 l1_scale_coeff=100):
        super().__init__()
        self.generator = Unet(generator_n_layers, in_channels, inner_channels,
                              out_channels, start_num_filters)
        self.descriminator = PatchGAN(descriminator_n_layers, in_channels + out_channels,
                                      start_num_filters)
        self.loss = Loss()
        self.l1_scale_coeff = l1_scale_coeff

    def forward(self, input=None, true_output=None, G_step=False):
        # input is a list of [model input, G(model input)]
        L, generated_img = input
        full_generated_img = torch.cat([L, generated_img], dim=1)
        if not G_step:
            desc_out_generated = self.descriminator(full_generated_img.detach())
            desc_ce_generated = self.loss(desc_out_generated, False)

            true_img = torch.cat([L, true_output], dim=1)
            desc_out_true = self.descriminator(true_img)
            desc_ce_true = self.loss(desc_out_true, True)

            desc_loss = (desc_ce_generated + desc_ce_true) / 2
            return desc_loss, desc_ce_generated, desc_ce_true
        else:
            desc_out_generated = self.descriminator(full_generated_img)
            generator_ce_loss, generator_l1_loss = \
                self.loss([desc_out_generated, generated_img],
                          true_output, G_loss=True)
            generator_loss = generator_ce_loss + generator_l1_loss * self.l1_scale_coeff
            return [generator_loss, generator_ce_loss, generator_l1_loss]
