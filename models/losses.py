import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):

        mask = predicted > 1e-3

        g = torch.log(predicted[mask]) - torch.log(target[mask])

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class MIMOUnetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [SILogLoss()]
        self.factors = [1]

    def forward(self, prediction, target):

        predicted_1, predicted_2, predicted_4 = prediction

        loss_accumulated = 0
        target_2 = F.interpolate(target, scale_factor=0.5, recompute_scale_factor=True)
        target_4 = F.interpolate(target, scale_factor=0.25, recompute_scale_factor=True)

        for factor, loss in zip(self.factors, self.losses):
            loss_accumulated += (
                (
                    loss(predicted_1, target)
                    + loss(predicted_2, target_2)
                    + loss(predicted_4, target_4)
                )
                / 3
                * factor
            )

        return loss_accumulated


class LadderNetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [SILogLoss()]
        self.factors = [1]

    def forward(self, prediction, target):

        predicted_1, predicted_8, predicted_16, *_ = prediction

        loss_accumulated = 0
        target_8 = F.interpolate(
            target, scale_factor=1 / 8, recompute_scale_factor=True
        )
        target_16 = F.interpolate(
            target, scale_factor=1 / 16, recompute_scale_factor=True
        )

        for factor, loss in zip(self.factors, self.losses):
            loss_accumulated += (
                (
                    loss(predicted_1, target)
                    + loss(predicted_8, target_8)
                    + loss(predicted_16, target_16)
                )
                / 3
                * factor
            )

        return loss_accumulated
