import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):

        mask = predicted > 1e-3

        g = torch.log(predicted[mask]) - torch.log(target[mask])

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [SILogLoss(), nn.L1Loss(reduction="mean")]
        self.factors = [1, 3]

    def forward(self, predicted, target):
        loss_accumulated = 0
        for factor, loss in zip(self.factors, self.losses):
            loss_accumulated += loss(predicted, target) * factor

        return loss_accumulated
