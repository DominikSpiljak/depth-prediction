import csv
from pathlib import Path

import cv2 as cv
import numpy as np
from PIL import Image


class DataNYUDepthLoader:
    def __init__(self, filepath, split):
        self.rgb_paths = []
        self.depth_paths = []

        if split == "val":
            split = "test"

        self.split = split

        with (Path(filepath) / "data" / f"nyu2_{split}.csv").open() as inp:
            reader = csv.reader(inp)
            for line in reader:
                rgb_path, depth_path = line
                self.rgb_paths.append(str(Path(filepath) / Path(rgb_path)))
                self.depth_paths.append(str(Path(filepath) / Path(depth_path)))

    def __getitem__(self, key):
        rgb_image = (
            np.asarray(Image.open(self.rgb_paths[key])).reshape((480, 640, 3))
            / 255  # type:ignore
        ).astype(np.float32)
        depth_map = (
            np.asarray(Image.open(self.depth_paths[key])).reshape(  # type:ignore
                (480, 640, 1)
            )
        ).astype(np.float32)

        if self.split == "train":
            depth_map = depth_map / 255 * 10
        else:
            depth_map = depth_map / 1000
        rgb_image = rgb_image[16 : 480 - 16, 16 : 640 - 16, :]
        depth_map = depth_map[16 : 480 - 16, 16 : 640 - 16, :]

        return (
            self.rgb_paths[key],
            self.depth_paths[key],
            rgb_image,
            depth_map,
        )

    def __len__(self):
        return len(self.rgb_paths)


class DataNYUDepthLoaderEigen:
    def __init__(self, filepath):
        self.eigen_root = Path(filepath) / "eigen_nyu_test"
        crop = np.load(str(self.eigen_root / "eigen_test_crop.npy"))

        rgb_images = np.load(str(self.eigen_root / "eigen_test_rgb.npy"))
        self.rgb_images = rgb_images[:, crop[0] : crop[1], crop[2] : crop[3], :]

        depth_images = np.load(str(self.eigen_root / "eigen_test_depth.npy"))
        self.depth_images = depth_images[:, crop[0] : crop[1], crop[2] : crop[3]]

    def __getitem__(self, key):
        return (
            str(self.eigen_root / "eigen_test_rgb.npy"),
            str(self.eigen_root / "eigen_test_depth.npy"),
            self.rgb_images[key],
            self.depth_images[key, ..., np.newaxis],
        )

    def __len__(self):
        return len(self.rgb_images)


class DataCityScapesLoader:
    def __init__(self, filepath, split):
        self.rgb_paths = []
        self.depth_paths = []

        rgb_root = Path(filepath) / "leftImg8bit_trainvaltest" / "leftImg8bit" / split
        depth_root = Path(filepath) / "disparity_trainvaltest" / "disparity" / split

        for image in rgb_root.glob("**/*.png"):
            self.rgb_paths.append(str(image))
            self.depth_paths.append(
                str(
                    depth_root
                    / image.parent.name
                    / image.name.replace("leftImg8bit", "disparity")
                )
            )

    def __getitem__(self, key):
        rgb_image = cv.imread(self.rgb_paths[key])
        disparity_image = cv.imread(self.depth_paths[key], cv.IMREAD_UNCHANGED).astype(
            np.float32
        )
        disparity_image[disparity_image > 0] = (
            disparity_image[disparity_image > 0] - 1
        ) / 256

        return (
            self.rgb_paths[key],
            self.depth_paths[key],
            rgb_image,
            disparity_image[..., np.newaxis],
        )

    def __len__(self):
        return len(self.rgb_paths)


if __name__ == "__main__":
    loader = DataNYUDepthLoaderEigen("/home/aromaticconfusion/datasets/NYU-depth/")

    for i in range(len(loader)):
        print(loader[i][3].shape)
