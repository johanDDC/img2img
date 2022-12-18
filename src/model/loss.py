import torch
import torch.nn as nn
import torch.nn.init as init


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.CE_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predictions, targets, G_loss=False):
        if G_loss:
            # predictions is a list of [discriminator out on generated img, generated img]
            # targets is ground truth
            true_target = torch.ones_like(predictions[0], dtype=torch.float32,
                                          device=predictions[0].device)
            return self.CE_loss(predictions[0], true_target), \
                   self.l1_loss(predictions[1], targets)
        else:
            # target is True, if predictions is not generated
            # and False otherwise
            targets = torch.full_like(predictions, float(targets),
                                      dtype=torch.float32, device=predictions.device)
            return self.CE_loss(predictions, targets)
