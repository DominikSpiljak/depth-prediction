import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self, variance_focus=0.85):
        super().__init__()
        self.variance_focus = variance_focus

    def forward(self, predicted, target):

        mask = predicted > 1e-3

        d = torch.log(predicted[mask]) - torch.log(target[mask])

        return (
            torch.sqrt((d**2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        )


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [SILogLoss(), nn.L1Loss(reduction="mean")]
        self.factors = [1, 0.5]

    def forward(self, predicted, target):
        loss_accumulated = 0
        for factor, loss in zip(self.factors, self.losses):
            loss_accumulated += loss(predicted, target) * factor

        return loss_accumulated
