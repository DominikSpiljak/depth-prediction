import pytorch_lightning as pl

from depth_mimo_unet_model.depth_mimo_unet import MIMOUnet
from depth_mimo_unet_model.losses import SILogLoss


class DepthMIMOUnetModule(pl.LightningModule):
    def __init__(
        self,
        *,
        data_args,
        model_args,
        training_args,
        logger_args,
    ):
        super().__init__()

        if not isinstance(model_args, Namespace):
            model_args = Namespace(**model_args)

        self.data_args = data_args
        self.model_args = model_args
        self.training_args = training_args
        self.logger_args = logger_args
        self.model = MIMOUnet(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )
        self.criterion = SILogLoss()

    def training_step(self, batch):
        indices, rgb_im, depth_map = batch

        predictions = self.model(rgb_im)

        loss = self.criterion(predicted=predictions, target=target)

        # TODO: Metrics

        return loss

    def validation_step(self, batch):
        indices, rgb_im, depth_map = batch

        predictions = self.model(rgb_im)

        # TODO: Metrics

    def test_step(self, batch):
        indices, rgb_im, depth_map = batch

        predictions = self.model(rgb_im)

        # TODO: Metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.training_args.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=3, min_lr=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "Validation loss",
        }
