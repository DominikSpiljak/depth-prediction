import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        g = torch.log(predicted * 5 + 5) - torch.log(target * 5 + 5)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
