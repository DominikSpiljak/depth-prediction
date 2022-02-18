import h5py
import numpy as np


class DataMatLoader:
    def __init__(self, filepath):
        self.file_handle = h5py.File(filepath)

    def __getitem__(self, key):
        rgb_image = np.rot90(
            np.moveaxis(np.array(self.file_handle.get("images")[key]), 0, -1), k=3
        )
        depth_map = np.rot90(np.array(self.file_handle.get("depths")[key]), k=3)
        return rgb_image, depth_map

    def __len__(self):
        return len(self.file_handle.get("images"))
