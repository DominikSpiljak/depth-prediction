from pathlib import Path

import cv2 as cv
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms

from argument_parser import parse_args
from data.data_module import DepthEstimationDataModule
from data.loader import DataMatLoader
from data.visualiser import visualise_depth
from models.depth_mimo_unet_module import DepthMIMOUnetModule


def main():
    args = parse_args()
    data_module = DepthEstimationDataModule(
        data_args=args.data, training_args=args.training
    )
    depth_mimounet_module = DepthMIMOUnetModule(
        model_args=args.model, training_args=args.training, logging_args=args.logging
    )

    metric_monitor_callbacks = [
        ModelCheckpoint(
            save_last=True,
            verbose=True,
            monitor="Validation loss",
            save_top_k=args.logging.save_top_k,
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        callbacks=metric_monitor_callbacks,
        gpus=-1,
    )

    if not args.training.eval_mode:
        trainer.fit(depth_mimounet_module, data_module)

    trainer.test(depth_mimounet_module, data_module)


if __name__ == "__main__":
    main()
