import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from data.augmentations import get_augmentations
from data.loader import DataNYUDepthLoader, DataCityScapesLoader


class DepthEstimationDataset(Dataset):
    def __init__(
        self,
        loader,
        base_transformation,
        rgb_normalization,
        indices=None,
        augmentations=None,
    ):
        self.loader = loader
        if indices is None:
            self.indices = list(range(len(loader)))
        else:
            self.indices = indices
        self.base_transformation = base_transformation
        self.rgb_normalization = rgb_normalization
        self.augmentations = augmentations

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rgb_path, depth_path, rgb_image, depth_map = self.loader[self.indices[idx]]
        rgb_image = self.base_transformation(rgb_image.copy())
        depth_map = self.base_transformation(depth_map.copy())

        if self.augmentations:
            rgb_image, depth_map = self.augmentations(rgb_image, depth_map)

        return (
            self.indices[idx],
            rgb_path,
            depth_path,
            self.rgb_normalization(rgb_image),
            depth_map,
        )


def collate_fn(batch):
    indices = [sample[0] for sample in batch]
    rgb_paths = [sample[1] for sample in batch]
    depth_paths = [sample[2] for sample in batch]
    rgb_images = torch.stack([sample[3] for sample in batch])
    depth_maps = torch.stack([sample[4] for sample in batch])
    return indices, rgb_paths, depth_paths, rgb_images, depth_maps


class DepthEstimationDataModule(pl.LightningDataModule):
    def __init__(self, *, data_args, training_args):
        self.data_path = data_args.dataset
        self.batch_size = training_args.batch_size
        self.num_workers = training_args.num_workers
        self.base_transformation = [
            transforms.ToTensor(),
            transforms.Resize(list(map(int, data_args.image_size))),
        ]
        self.rgb_normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.augmentations = get_augmentations(data_args)

        self.val_ratio = training_args.val_ratio
        self.test_ratio = training_args.test_ratio

    def setup(self, stage):
        if self.data_path.name == "NYU-depth":
            self.__setup_nyu_depth(stage)

        elif self.data_path.name == "Cityscapes":
            self.__setup_cityscapes_depth(stage)

    def __setup_nyu_depth(self, stage):
        loader = DataNYUDepthLoader(self.data_path)

        num_samples = len(loader)
        num_val_samples = int(num_samples * self.val_ratio)
        num_test_samples = int(num_samples * self.test_ratio)

        num_train_samples_ = num_samples - num_val_samples
        train_indices_, val_indices = random_split(
            torch.arange(num_samples),
            [num_train_samples_, num_val_samples],
            generator=torch.Generator().manual_seed(42),
        )

        num_train_samples = num_train_samples_ - num_test_samples
        train_indices, test_indices = random_split(
            train_indices_,
            [num_train_samples, num_test_samples],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = DepthEstimationDataset(
            loader=loader,
            indices=train_indices,
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
            augmentations=self.augmentations,
        )
        self.val_dataset = DepthEstimationDataset(
            loader=loader,
            indices=val_indices,
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
        )
        self.test_dataset = DepthEstimationDataset(
            loader=loader,
            indices=test_indices,
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
        )

    def __setup_cityscapes_depth(self, stage):
        self.train_dataset = DepthEstimationDataset(
            loader=DataCityScapesLoader(self.data_path, "train"),
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
            augmentations=self.augmentations,
        )
        self.val_dataset = DepthEstimationDataset(
            loader=DataCityScapesLoader(self.data_path, "val"),
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
        )
        self.test_dataset = DepthEstimationDataset(
            loader=DataCityScapesLoader(self.data_path, "test"),
            base_transformation=transforms.Compose(self.base_transformation),
            rgb_normalization=self.rgb_normalization,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )
