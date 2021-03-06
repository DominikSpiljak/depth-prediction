from models.depth_estimation_module import DepthEstimationModule
from models.depth_mimo_unet_model.depth_mimo_unet import MIMOUnet
from models.losses import MIMOUnetCriterion


class DepthMIMOUnetModule(DepthEstimationModule):
    def get_model(self):
        return MIMOUnet(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )

    def get_criterion(self):
        return MIMOUnetCriterion()

    def get_final_prediction(self, prediction):
        return prediction[0]
