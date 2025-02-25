from abc import ABC, abstractmethod
import torch

class Dataset(ABC):
    @abstractmethod
    def load_train_data(
        self,
        batch_size: int,
        num_workers: int,
        validation: bool,
        train_indices: torch.Tensor,
        valid_indices: torch.Tensor,
    ) -> torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def load_test_data(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> torch.utils.data.DataLoader:
        pass