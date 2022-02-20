import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from data.loader import DataMatLoader


class DepthEstimationDataset(Dataset):
    def __init__(self, loader, indices, rgb_transformation, depth_transformation):
        self.loader = loader
        self.indices = indices
        self.rgb_transformation = rgb_transformation
        self.depth_transformation = depth_transformation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        rgb_image, depth_map = self.loader[self.indices[idx]]
        return (
            self.indices[idx],
            self.rgb_transformation(rgb_image.copy()),
            self.depth_transformation(depth_map.copy()),
        )


def collate_fn(batch):
    indices = [sample[0] for sample in batch]
    rgb_images = torch.stack([sample[1] for sample in batch])
    depth_maps = torch.stack([sample[2] for sample in batch])
    return indices, rgb_images, depth_maps


class DepthEstimationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        batch_size: int = 2,
        test_ratio: float = 0.2,
        val_ratio=0.1,
        num_workers=4,
        image_size=[240, 320],
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        base_transform = [transforms.ToTensor(), transforms.Resize(image_size)]
        self.rgb_transform = [
            *base_transform,
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        self.depth_transform = [
            *base_transform,
            transforms.Normalize((5), (5)),
        ]
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage):
        loader = DataMatLoader(self.data_path)

        num_samples = len(loader)
        num_val_samples = int(num_samples * self.val_ratio)
        num_test_samples = int(num_samples * self.test_ratio)
        indices = torch.arange(num_samples)

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
            rgb_transformation=transforms.Compose(self.rgb_transform),
            depth_transformation=transforms.Compose(self.depth_transform),
        )
        self.val_dataset = DepthEstimationDataset(
            loader=loader,
            indices=val_indices,
            rgb_transformation=transforms.Compose(self.rgb_transform),
            depth_transformation=transforms.Compose(self.depth_transform),
        )
        self.test_dataset = DepthEstimationDataset(
            loader=loader,
            indices=test_indices,
            rgb_transformation=transforms.Compose(self.rgb_transform),
            depth_transformation=transforms.Compose(self.depth_transform),
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
