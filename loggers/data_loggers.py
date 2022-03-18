import numpy as np
from data.visualiser import visualise_depth


class ImageLogger:
    def __init__(self, max_batches_logged_per_epoch, prefix):
        self.max_batches_logged_per_epoch = max_batches_logged_per_epoch
        self.prefix = prefix
        self.current_log = []

    def __call__(self, outputs):
        for index, _ in enumerate(outputs["indices"]):
            if len(self.current_log) >= self.max_batches_logged_per_epoch:
                return

            prediction = np.moveaxis(outputs["predictions"][index].numpy(), 0, -1)
            depth_map = np.moveaxis(outputs["depth_maps"][index].numpy(), 0, -1)
            rgb_im = np.moveaxis(outputs["rgb_im"][index].numpy(), 0, -1)

            self.current_log.append(
                visualise_depth(
                    depth_map=depth_map, rgb_im=rgb_im, prediction=prediction
                )
            )

    def compute(self, epoch, logger):
        images = np.vstack(self.current_log)
        logger.experiment.add_image(
            f"{self.prefix}/epoch={epoch}", np.moveaxis(images, -1, 0)
        )
        self.current_log = []
