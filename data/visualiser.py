import cv2 as cv
import numpy as np


def visualise_depth(
    *, depth_map, rgb_im=None, prediction=None, normalized=True, imagenet_norm=False
):

    if prediction is not None:
        depth_map = np.hstack((prediction, depth_map))

    min_val = depth_map.min()
    max_val = depth_map.max()
    if min_val == max_val:
        shifted_map = np.zeros_like(depth_map)
    else:
        shifted_map = (depth_map - min_val) / (max_val - min_val)

    # Invert color map because it is more intuitive
    shifted_map = shifted_map * -1 + 1
    depth_color_map = cv.applyColorMap(
        (shifted_map * 255).astype(np.uint8), cv.COLORMAP_MAGMA
    )
    if rgb_im is None:
        return depth_color_map

    if normalized:
        if imagenet_norm:
            rgb_im = rgb_im * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )
        else:
            rgb_im = (rgb_im + 1) / 2

    rgb_im = (rgb_im * 255).astype(np.uint8)

    return np.hstack(
        (
            rgb_im,
            cv.cvtColor(
                depth_color_map,
                cv.COLOR_BGR2RGB,
            ),
        )
    )
