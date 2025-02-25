import os
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from .Dataset import Dataset
from .transforms import Cutout

class CIFAR10(Dataset):
    def __init__(self, train_indices: torch.Tensor = None, valid_indices: torch.Tensor = None):
        self.train_indices = train_indices
        self.valid_indices = valid_indices

    def load_train_data(
        self,
        batch_size: int,
        num_workers: int,
        validation: bool,
    ) -> DataLoader:
        if not validation:
            assert self.valid_indices is None

        if self.train_indices is not None or self.valid_indices is not None:
            assert self.train_indices is not None and self.valid_indices is not None

        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                Cutout(1, length=8),
            ]
        )
        dataset = DataLoader(
            torchvision.datasets.CIFAR10(
                root=os.path.join(os.getcwd(), "data"),
                train=True,
                download=True,
                transform=transform,
            )
        )
        if validation:
            train_size = int(0.9 * len(dataset))  # Use 45,000 samples for training.
            valid_size = int(0.1 * len(dataset))  # Use  5,000 samples for validation.
        else:
            train_size = int(len(dataset))
            valid_size = 0

        if valid_size > 0:
            if self.train_indices is None:
                train_dataset, valid_dataset = torch.utils.data.random_split(
                    dataset.dataset, [train_size, valid_size]
                )
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    train_dataset.indices
                )
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    valid_dataset.indices
                )
            else:
                train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_indices)
                valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.valid_indices)

            valid_loader = DataLoader(
                dataset.dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                sampler=valid_sampler,
            )
        else:
            train_sampler = None
            valid_loader = None

        train_loader = DataLoader(
            dataset.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        if validation:
            return valid_loader
        else:
            return train_loader
    
    def load_test_data(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        test_loader = DataLoader(
        dataset=torchvision.datasets.CIFAR10(
                root=os.path.join(os.getcwd(), "data"),
                train=False,
                download=True,
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        return test_loader