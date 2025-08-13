from abc import ABC, abstractmethod
import torch
import os
import logging
from aihwkit.nn import AnalogConv2d, AnalogLinear
from lionheart.datasets.Dataset import Dataset
from lionheart.models.Model import Model

class TrainerEvaluator(ABC):
    def __init__(self):
        self.dataset = self.instantiate_dataset()
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def digital_analog_parameters(self, model: Model = None):
        if model is None:
            model = self.model

        digital_parameters = list(model.parameters())
        analog_parameters = []
        for m in model.modules():
            if isinstance(m, (AnalogLinear, AnalogConv2d)):
                analog_parameters.extend(m.parameters())

        digital_parameters = list(set(digital_parameters) - set(analog_parameters))
        return digital_parameters, analog_parameters
        
    def save_checkpoint(self, checkpoint_path: str, ind_analog_layers: list[int] = None):
        assert self.model is not None
        self.model.convert_layers_to_digital()
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
                "scheduler": self.scheduler, 
                "ind_analog_layers": ind_analog_layers,
            },
            checkpoint_path
        )
        self.model.convert_layers_to_analog(ind_analog_layers)

    def load_checkpoint(self, checkpoint_path: str, ind_analog_layers: list[int] = None):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            try:
                if checkpoint["optimizer"] is not None:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
            except:
                logging.warning("failed to load optimizer.")

            self.scheduler = checkpoint["scheduler"]
            self.set_model()
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.convert_layers_to_analog(checkpoint['ind_analog_layers'] if ind_analog_layers is None else ind_analog_layers)
            return checkpoint['ind_analog_layers'] if ind_analog_layers is None else ind_analog_layers
        else:
            logging.warning('checkpoint_path does not exist.')
            return None

            

    @abstractmethod
    def instantiate_model(self) -> Model:
        pass

    def set_model(self, eval: bool = False):
        self.model = self.instantiate_model()
        self.model.ind_analog_layers = []
        if eval:
            self.model.eval()

    @abstractmethod
    def instantiate_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def instantiate_optimizer(self, digital_lr: float, digital_momentum: float, analog_lr: float, analog_momentum: float) -> torch.optim.Optimizer:
        pass

    def set_optimizer(self, digital_lr: float, digital_momentum: float, analog_lr: float, analog_momentum: float):
        self.optimizer = self.instantiate_optimizer(digital_lr=digital_lr, digital_momentum=digital_momentum, analog_lr=analog_lr, analog_momentum=analog_momentum)
    
    @abstractmethod
    def instantiate_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        pass

    def set_scheduler(self):
        self.scheduler = self.instantiate_scheduler()

    @abstractmethod
    def train(self, num_steps: int, batch_size: int, num_workers: int, logging_freq: int):
        pass

    @abstractmethod
    def evaluate(self, batch_size: int, num_workers: int):
        pass