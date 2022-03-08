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
        self.losses = [SILogLoss()]

    def forward(self, predicted, target):
        loss_accumulated = 0
        for loss in self.losses:
            loss_accumulated += loss(predicted, target)

        return loss_accumulated / len(self.losses)
