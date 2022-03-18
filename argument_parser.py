from argparse import ArgumentParser, Namespace
from pathlib import Path


class DotDict(dict):
    # Simple dict where values can be accessed with dot annotation
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


def parse_args():
    parser = ArgumentParser(
        prog="Deblurring math expressions",
        description="Script that trains/evaluates/exploits model for deblurring",
    )

    data = parser.add_argument_group("data")
    training = parser.add_argument_group("training")
    model = parser.add_argument_group("model")
    logging = parser.add_argument_group("logging")

    data.add_argument(
        "--dataset",
        help="Path to .mat dataset file",
        type=Path,
        default=Path.home() / "datasets/NYU-depth/nyu_depth_v2_labeled.mat",
    )
    data.add_argument(
        "--image-size",
        help="Image size used for training, split with comma",
        nargs=2,
        default=[240, 320],
    )

    data.add_argument(
        "--aug-gaussian-blur",
        help="Wether to use Gaussian blur as one of the augmentations",
        action="store_true",
    )

    data.add_argument(
        "--aug-horizontal-flip",
        help="Wether to use Horizontal flip as one of the augmentations",
        action="store_true",
    )

    data.add_argument(
        "--aug-vertical-flip",
        help="Wether to use Vertical flip as one of the augmentations",
        action="store_true",
    )

    data.add_argument(
        "--aug-adjust-gamma",
        help="Wether to use Gamma adjusting as one of the augmentations",
        action="store_true",
    )

    data.add_argument(
        "--aug-adjust-hue",
        help="Wether to use Hue adjusting as one of the augmentations",
        action="store_true",
    )

    data.add_argument(
        "--aug-adjust-contrast",
        help="Wether to use Contrast adjusting as one of the augmentations",
        action="store_true",
    )

    model.add_argument(
        "--num-res-blocks",
        help="Number of ResNet blocks inside MIMOUnet block",
        default=8,
        type=int,
    )

    training.add_argument(
        "--val-ratio",
        help="Percentage of data to be used as validation dataset",
        type=float,
        default=0.1,
    )
    training.add_argument(
        "--test-ratio",
        help="Percentage of data to be used as test dataset",
        type=float,
        default=0.2,
    )
    training.add_argument(
        "--learning-rate", help="Learning rate of the model", type=float, default=1e-4
    )
    training.add_argument(
        "--batch-size", help="Batch size for training", type=int, default=2
    )
    training.add_argument(
        "--num-workers", help="Number of workers", type=int, default=4
    )
    training.add_argument(
        "--gpus", help="Number of gpus used for training", type=int, default=-1
    )
    training.add_argument(
        "--eval-mode",
        help="Enables evaluation mode for a checkpoint",
        action="store_true",
    )
    training.add_argument("--checkpoint", help="Path to checkpoint", type=Path)

    logging.add_argument(
        "--save-top-k", help="Top k checkpoints to save", type=int, default=1
    )
    logging.add_argument(
        "--max-images-logged-per-epoch",
        help="Number of images logged per epoch",
        type=int,
        default=10,
    )
    logging.add_argument(
        "--disable-image-logging",
        help="Wether to disable image logging",
        action="store_true",
    )
    logging.add_argument(
        "--disable-metric-collection",
        help="Wether to metric logging",
        action="store_true",
    )

    args = parser.parse_args()

    # Transform the Namespace into a dict with groups
    arg_groups = DotDict()

    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        arg_groups[group.title] = Namespace(**group_dict)

    return arg_groups
