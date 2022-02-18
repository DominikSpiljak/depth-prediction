import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from data.loader import DataMatLoader


class DepthEstimationDataset(Dataset):
    def __init__(self, loader, indices, transformations):
        self.loader = loader
        self.indices = indices
        self.transformations = transformations

    def __len__(self):
        return len(indices)

    def __getitem__(self, idx):
        rgb_image, depth_map = self.loader[self.indices[idx]]
        return transformations(rgb_image), transformations(depth_map)


def collate_fn(self, batch):
    rgb_images = torch.stack([sample[0] for sample in batch])
    depth_maps = torch.stack([sample[1] for sample in batch])
    return rgb_images, depth_maps


class DepthEstimationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        data_path: str,
        batch_size: int = 2,
        test_ratio: float = 0.2,
        val_ratio=0.1,
        image_size=[240, 320],
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(image_size)]
        )
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def setup(self, stage):
        loader = DataMatLoader(data_path)

        num_samples = len(loader)
        num_val_samples = int(num_samples * val_ratio)
        num_test_samples = int(num_samples * test_ratio)
        indices = torch.arange(num_samples)

        num_train_samples_ = num_samples - num_val_samples
        train_indices_, val_indices = random_split(
            torch.arange(num_samples),
            [num_train_samples_, num_val_samples],
            generator=torch.Generator().manual_seed(42),
        )

        num_train_samples = num_train_samples_ - num_val_samples
        train_indices, test_indices = random_split(
            torch.arange(train_indices_),
            [num_train_samples, num_test_samples],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dataset = DepthEstimationDataset(
            loader=loader, indices=train_indices, transformation=self.transform
        )
        self.val_dataset = DepthEstimationDataset(
            loader=loader, indices=val_indices, transformation=self.transform
        )
        self.test_dataset = DepthEstimationDataset(
            loader=loader, indices=test_indices, transformation=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            pin_memory=True,
        )
