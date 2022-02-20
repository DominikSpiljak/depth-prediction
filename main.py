from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms

from data.data_module import DepthEstimationDataModule
from data.loader import DataMatLoader
from data.visualiser import visualise_depth

from models.depth_mimo_unet_module import DepthMIMOUnetModule


def main():
    DATASET = Path.home() / "datasets/NYU-depth/nyu_depth_v2_labeled.mat"
    data_module = DepthEstimationDataModule(data_path=str(DATASET))
    depth_mimounet_module = DepthMIMOUnetModule()


if __name__ == "__main__":
    main()