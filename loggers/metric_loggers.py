import torch


class Metric:
    def __init__(self, prefix):
        self.prefix = prefix
        self.metric_name = None
        self.aggregated = 0
        self.num_steps = 0

    def __call__(self, outputs):
        target = outputs["depth_maps"]
        pred = outputs["predictions"]
        self.aggregated += self.forward(pred, target)
        self.num_steps += 1

    def forward(self, pred, target):
        raise NotImplementedError("Forward not implemented for base metric")

    def compute(self, epoch, logger):
        if self.metric_name is None:
            raise NotImplementedError("Metric name needs to be specified")
        logger.experiment.add_scalars(
            f"{self.prefix} {self.metric_name}",
            {self.metric_name: self.aggregated / self.num_steps},
            global_step=epoch,
        )
        self.aggregated = 0
        self.num_steps = 0


class DeltaError(Metric):
    def __init__(self, prefix, less_than_value=1.25, exponent=1):
        super().__init__(prefix)
        self.metric_name = f"delta < {less_than_value} ^ {exponent}"
        self.less_than_value = less_than_value
        self.exponent = exponent

    def forward(self, pred, target):
        thresh = torch.max((target / pred), (pred / target))
        return (thresh < self.less_than_value ** self.exponent).float().mean()


class RelativeAbsoluteError(Metric):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.metric_name = "Relative Absolute Error"

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target) / target, dim=(2, 3)).mean()


class RelativeSquaredError(Metric):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.metric_name = "Relative Squared Error"

    def forward(self, pred, target):
        return torch.mean(torch.pow(pred - target, 2) / target, dim=(2, 3)).mean()


class RootMeanSquaredError(Metric):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.metric_name = "Root Mean Squared Error"

    def forward(self, pred, target):
        return torch.sqrt(torch.mean(torch.pow(pred - target, 2), dim=(2, 3))).mean()


class LogRootMeanSquaredError(Metric):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.metric_name = "Log Root Mean Squared Error"

    def forward(self, pred, target):
        return torch.sqrt(
            torch.mean(torch.pow(torch.log(pred) - torch.log(target), 2), dim=(2, 3))
        ).mean()


class Log10Error(Metric):
    def __init__(self, prefix):
        super().__init__(prefix)
        self.metric_name = "Log10 Error"

    def forward(self, pred, target):
        return torch.mean(
            torch.abs(torch.log10(pred) - torch.log10(target)), dim=(2, 3)
        ).mean()
