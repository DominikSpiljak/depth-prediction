from pathlib import Path
import json
import h5py
import numpy as np
import cv2 as cv


class DataNYUDepthLoader:
    def __init__(self, filepath):
        self.file_handle = h5py.File(filepath)

    def __getitem__(self, key):
        rgb_image = np.rot90(
            np.moveaxis(np.array(self.file_handle.get("images")[key]), 0, -1), k=3
        )
        depth_map = np.rot90(np.array(self.file_handle.get("depths")[key]), k=3)
        return rgb_image, depth_map[..., np.newaxis]

    def __len__(self):
        return len(self.file_handle.get("images"))


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
    loader = DataCityScapesLoader(
        "/home/aromaticconfusion/datasets/Cityscapes/", "test"
    )
    print(len(loader))
