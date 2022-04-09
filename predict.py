from argument_parser import ArgumentParser

from data.visualiser import visualise_depth
import cv2 as cv
import numpy as np
import open3d as o3d
from torchvision import transforms
from models.depth_mimo_unet_module import DepthMIMOUnetModule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to checkpoint")
    parser.add_argument("--image", help="Path to image for predicting")
    parser.add_argument(
        "--image_size",
        help="Image size before entering the model",
        nargs=2,
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
    rgb_image = cv.cvtColor(rgb_image, cv.COLOR_RGB2BGR)

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
    if args._3d:
        rgbd_to_pointcloud(
            np.moveaxis(image[0].detach().numpy(), 0, -1) * 0.5 + 0.5,
            depth_map=depth[0].squeeze().detach().numpy() * 10,
        )
    else:
        visualised = visualise_depth(
            depth_map=np.moveaxis(depth[0].detach().numpy(), 0, -1),
            rgb_im=np.moveaxis(image[0].detach().numpy(), 0, -1),
        )
        cv.imshow("Depth map", visualised)
        cv.waitKey()


if __name__ == "__main__":
    main()
