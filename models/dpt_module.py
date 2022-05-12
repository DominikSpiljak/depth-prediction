from models.depth_estimation_module import DepthEstimationModule
from models.dpt_model.dense_prediction_transformer import DPT
from models.losses import DPTCriterion


class DPTModule(DepthEstimationModule):
    def get_model(self):
        return DPT(**{k: v for k, v in vars(self.model_args).items() if v is not None})

    def get_criterion(self):
        return DPTCriterion()

    def get_final_prediction(self, prediction):
        return prediction
