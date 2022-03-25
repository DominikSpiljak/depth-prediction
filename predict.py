from argument_parser import ArgumentParser

from data.visualiser import visualise_depth
import cv2 as cv
import numpy as np
from torchvision import transforms
from models.depth_mimo_unet_module import DepthMIMOUnetModule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--image", help="Path to image for predicting")
    parser.add_argument(
        "--image_size", help="Image size before entering the model", nargs=2
    )
    return parser.parse_args()


def main():
    args = parse_args()
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(list(map(int, args.image_size))),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model = DepthMIMOUnetModule.load_model_from_ckpt(
        args.checkpoint,
    )

    image = preprocess(cv.imread(args.image)).unsqueeze(0)
    depth, *_ = model(image)

    visualised = visualise_depth(
        depth_map=np.moveaxis(depth[0].detach().numpy(), 0, -1),
        rgb_im=np.moveaxis(image[0].detach().numpy(), 0, -1),
    )
    cv.imshow("Depth map", visualised)
    cv.waitKey()


if __name__ == "__main__":
    main()
