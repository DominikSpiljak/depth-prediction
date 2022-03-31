from pathlib import Path
import csv
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2 as cv


class DataNYUDepthLoader:
    def __init__(self, filepath, split, max_depth=1000.0):
        self.rgb_paths = []
        self.depth_paths = []
        self.max_depth = max_depth

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
            np.asarray(Image.open(self.rgb_paths[key]))  # type:ignore
            .reshape((480, 640, 3))
            / 255
        ).astype(np.float32)
        depth_map = (
            np.asarray(Image.open(self.depth_paths[key]))  # type:ignore
            .reshape((480, 640, 1))
        ).astype(np.float32)

        if self.split == "train":
            depth_map = depth_map / 255
        else:
            depth_map = depth_map / 1000 / 10
        rgb_image = rgb_image[16: 480 - 16, 16:640 - 16, :]
        depth_map = depth_map[16: 480 - 16, 16:640 - 16, :]

        return (
            self.rgb_paths[key],
            self.depth_paths[key],
            rgb_image,
            depth_map,
        )

    def __len__(self):
        return len(self.rgb_paths)


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
        # TODO: Fix red and blue shifted
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
    loader = DataNYUDepthLoader("/home/aromaticconfusion/datasets/NYU-depth/", "val")
    max = 0
    for i in tqdm(range(len(loader))):
        if loader[i][3].max() > max:
            max = loader[i][3].max()
    print(max)
