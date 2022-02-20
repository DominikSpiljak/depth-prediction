import torch


class SILogLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted, target):
        g = torch.log(predicted) - torch.log(target)

        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
