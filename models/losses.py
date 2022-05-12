import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):

        # mask = predicted > 0

        g = torch.log(predicted) - torch.log(target)

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
    def __init__(self, aux_weight=0.1):
        super().__init__()
        self.loss = SILogLoss()
        self.aux_loss = SILogLoss()
        self.aux_weight = aux_weight

    def forward(self, prediction, target):

        predicted_1, predicted_8, predicted_16, predicted_32, predicted_64 = prediction

        target_8 = F.interpolate(
            target, scale_factor=1 / 8, recompute_scale_factor=True
        )
        target_16 = F.interpolate(
            target, scale_factor=1 / 16, recompute_scale_factor=True
        )
        target_32 = F.interpolate(
            target, scale_factor=1 / 32, recompute_scale_factor=True
        )
        target_64 = F.interpolate(
            target, scale_factor=1 / 64, recompute_scale_factor=True
        )

        loss = self.loss(predicted=predicted_1, target=target)
        aux_loss = (
            self.aux_loss(predicted=predicted_8, target=target_8)
            + self.aux_loss(predicted=predicted_16, target=target_16)
            + self.aux_loss(predicted=predicted_32, target=target_32)
            + self.aux_loss(predicted=predicted_64, target=target_64) * 5
        ) / 8

        return loss + self.aux_weight * aux_loss


class DPTCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [SILogLoss()]
        self.factors = [1]

    def forward(self, prediction, target):
        loss_accumulated = 0

        for factor, loss in zip(self.factors, self.losses):
            loss_accumulated += loss(prediction, target) * factor

        return loss_accumulated
