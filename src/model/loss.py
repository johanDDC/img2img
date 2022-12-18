import torch
import torch.nn as nn
import torch.nn.init as init


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.CE_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def G_loss(self, discriminator_out, generated_img,
               ground_truth):
        true_target = torch.ones_like(discriminator_out, dtype=torch.float32,
                                      device=discriminator_out.device)
        return self.CE_loss(discriminator_out, true_target), \
                   self.l1_loss(generated_img, ground_truth)

    def D_loss(self, discriminator_out, mark):
        if mark:
            target = torch.ones_like(discriminator_out, dtype=torch.float32,
                                      device=discriminator_out.device)
        else:
            target = torch.zeros_like(discriminator_out, dtype=torch.float32,
                                     device=discriminator_out.device)
        return self.CE_loss(discriminator_out, target)

    def forward(self):
        raise NotImplementedError()