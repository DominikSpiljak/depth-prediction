from pathlib import Path

import cv2 as cv
import numpy as np

from data.data_module import DepthEstimationDataModule


def main():
    DATASET = Path.home() / "datasets/NYU-depth/nyu_depth_v2_labeled.mat"
    data_module = DepthEstimationDataModule(data_path=DATASET)


if __name__ == "__main__":
    main()