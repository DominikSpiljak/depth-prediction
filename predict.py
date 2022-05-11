import numpy as np
import open3d as o3d
from PIL import Image
from torchvision import transforms

from argument_parser import ArgumentParser
from data.visualiser import visualise_depth
from models.depth_mimo_unet_module import DepthMIMOUnetModule
from models.laddernet_module import LadderNetModule

module_mapping = {
    "MIMOUnet": DepthMIMOUnetModule,
    "LadderNet": LadderNetModule,
    "DPT": None,
}

model_names = ["MIMOUnet", "LadderNet", "DPT"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        help="Which model to use",
        choices=model_names,
        default=model_names[0],
    )
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--image", help="Path to image for predicting")
    parser.add_argument(
        "--image-size",
        help="Image size before entering the model",
        nargs=2,
    )
    parser.add_argument(
        "--imagenet-norm",
        help="Wether to use imagenet normalization",
        action="store_true",
    )
    parser.add_argument(
        "--3d",
        help="Wether to display point cloud in 3d",
        action="store_true",
        dest="_3d",
    )
    return parser.parse_args()


def rgbd_to_pointcloud(rgb_image, depth_map):
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_image = (rgb_image * 255).astype(np.uint8).copy()

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_image),
        depth=o3d.geometry.Image(depth_map.astype(np.float32)),
        convert_rgb_to_intensity=False,
    )
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        ),
    )
    point_cloud.transform(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    o3d.visualization.draw_geometries([point_cloud])


def main():
    args = parse_args()
    print(args.imagenet_norm)
    if args.imagenet_norm:
        normalization = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        normalization = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(list(map(int, args.image_size))),
            transforms.Normalize(*normalization),
        ]
    )

    model = module_mapping[args.model].load_model_from_ckpt(
        args.checkpoint,
    )

    image = preprocess(Image.open(args.image)).unsqueeze(0)
    depth, *_ = model(image)
    if args.imagenet_norm:
        image_denorm = np.moveaxis(image[0].detach().numpy(), 0, -1) * np.array(
            [0.229, 0.224, 0.225]
        ) + np.array([(0.485, 0.456, 0.406)])
    else:
        image_denorm = np.moveaxis(image[0].detach().numpy(), 0, -1) * 0.5 + 0.5
    if args._3d:
        rgbd_to_pointcloud(
            image_denorm,
            depth_map=depth[0].squeeze().detach().numpy(),
        )
    else:
        visualised = visualise_depth(
            depth_map=np.moveaxis(depth[0].detach().numpy(), 0, -1),
            rgb_im=np.moveaxis(image[0].detach().numpy(), 0, -1),
            imagenet_norm=args.imagenet_norm,
        )
        vis = Image.fromarray(np.uint8(visualised)).convert("RGB")
        vis.show()


if __name__ == "__main__":
    main()
