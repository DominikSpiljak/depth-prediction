from models.depth_estimation_module import DepthEstimationModule
from models.laddernet_model.laddernet import LadderNet
from models.losses import LadderNetCriterion


class LadderNetModule(DepthEstimationModule):
    def get_model(self):
        return LadderNet(
            **{k: v for k, v in vars(self.model_args).items() if v is not None}
        )

    def get_criterion(self):
        return LadderNetCriterion()

    def get_final_prediction(self, prediction):
        return prediction[0]
