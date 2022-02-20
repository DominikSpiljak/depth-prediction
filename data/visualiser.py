import cv2 as cv
import numpy as np


def visualise_depth(*, depth_map, rgb_im=None, prediction=None, normalized=True):
    if normalized:
        depth_map = (depth_map + 1) / 2

    if prediction is not None:
        if normalized:
            prediction = (depth_map + 1) / 2

        depth_map = np.hstack((prediction, depth_map))

    min_val = depth_map.min()
    max_val = depth_map.max()
    if min_val == max_val:
        shifted_map = np.zeros_like(depth_map)
    else:
        shifted_map = (depth_map - min_val) / (max_val - min_val)
    depth_color_map = cv.applyColorMap(
        (shifted_map * 255).astype(np.uint8), cv.COLORMAP_JET
    )
    if rgb_im is None:
        return depth_color_map

    if normalized:
        rgb_im = (rgb_im + 1) / 2
    rgb_im = (rgb_im * 255).astype(np.uint8)

    return np.hstack((rgb_im, depth_color_map))
