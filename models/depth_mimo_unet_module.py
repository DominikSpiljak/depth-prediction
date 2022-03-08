from argparse import Namespace

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from loggers.data_loggers import ImageLogger
from models.depth_mimo_unet_model.depth_mimo_unet import MIMOUnet
from models.depth_mimo_unet_model.losses import Criterion


class DepthMIMOUnetModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_args,
        training_args,
        logging_args,
    ):
        super().__init__()

        if not isinstance(model_args, Namespace):
            model_args = Namespace(**model_args)

        self.model_args = model_args
        self.training_args = training_args
        self.logging_args = logging_args
        self.model = MIMOUnet(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )
        self.criterion = Criterion()
        self.setup_loggers()

    def setup_loggers(self):
        self.train_loggers = []
        self.validation_loggers = []
        self.test_loggers = []

        if not self.logging_args.disable_image_logging:
            self.validation_loggers.append(
                ImageLogger(
                    self.logging_args.max_images_logged_per_epoch,
                    "Validation",
                )
            )
            self.test_loggers.append(
                ImageLogger(
                    self.logging_args.max_images_logged_per_epoch,
                    "Test",
                )
            )

    def log_metrics(self, loggers, outputs):
        for logger in loggers:
            logger(outputs)

    def compute_loggers(self, loggers, epoch, logger):
        for logger_ in loggers:
            logger_.compute(epoch, logger)

    def forward(self, batch, *args, **kwargs):
        indices, rgb_im, depth_map = batch

        prediction, prediction_2, prediction_4 = self.model(rgb_im)

        depth_map_2 = F.interpolate(depth_map, scale_factor=0.5)
        depth_map_4 = F.interpolate(depth_map, scale_factor=0.25)

        loss = (
            self.criterion(predicted=prediction, target=depth_map)
            + self.criterion(predicted=prediction_2, target=depth_map_2)
            + self.criterion(predicted=prediction_4, target=depth_map_4)
        ) / 3

        return (
            indices,
            rgb_im.cpu(),
            prediction.detach().cpu(),
            depth_map.cpu(),
            loss,
        )

    def training_step(self, batch, *args, **kwargs):
        indices, rgb_im, prediction, depth_map, loss = self.forward(batch)

        # TODO: Metrics

        return {
            "indices": indices,
            "loss": loss,
            "rgb_im": rgb_im,
            "predictions": prediction,
            "depth_maps": depth_map,
        }

    def validation_step(self, batch, *args, **kwargs):
        indices, rgb_im, prediction, depth_map, loss = self.forward(batch)

        # TODO: Metrics

        return {
            "indices": indices,
            "loss": loss,
            "rgb_im": rgb_im,
            "predictions": prediction,
            "depth_maps": depth_map,
        }

    def test_step(self, batch, *args, **kwargs):
        indices, rgb_im, prediction, depth_map, loss = self.forward(batch)

        # TODO: Metrics

        return {
            "indices": indices,
            "loss": loss,
            "rgb_im": rgb_im,
            "predictions": prediction,
            "depth_maps": depth_map,
        }

    def training_step_end(self, outputs):
        self.log_metrics(self.validation_loggers, outputs)
        loss = outputs["loss"]
        self.log("Validation loss", loss, batch_size=self.training_args.batch_size)
        return loss

    def on_train_epoch_end(self):
        self.compute_loggers(self.train_loggers, self.current_epoch, self.logger)

    def validation_step_end(self, outputs):
        self.log_metrics(self.validation_loggers, outputs)
        loss = outputs["loss"]
        self.log("Validation loss", loss, batch_size=self.training_args.batch_size)
        return loss

    def on_validation_epoch_end(self):
        self.compute_loggers(self.validation_loggers, self.current_epoch, self.logger)

    def test_step_end(self, outputs):
        self.log_metrics(self.test_loggers, outputs)
        loss = outputs["loss"]
        self.log("Test loss", loss, batch_size=self.training_args.batch_size)
        return loss

    def on_test_epoch_end(self):
        self.compute_loggers(self.test_loggers, self.current_epoch, self.logger)

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
