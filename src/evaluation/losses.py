import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction, target, mask):
        # Cast everything to Float32 to prevent underflow
        prediction = prediction.float()
        target = target.float()
        mask = mask.float()

        loss = (prediction - target) ** 2
        loss = loss * mask

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss

        # Add an epsilon (1e-7) to prevent dividing by 0 if a patch has no valid pixels
        return loss.sum() / (mask.sum().clamp(min=1.0) + 1e-7)
