import logging
import torch
from aihwkit.optim import AnalogSGD
from .TrainerEvaluator import TrainerEvaluator
from lionheart.models.Model import Model
from lionheart.models.ResNet20 import ResNet20
from lionheart.models.ResNet8 import ResNet8
from lionheart.datasets import CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet20CIFAR10(TrainerEvaluator):
    def instantiate_model(self):
        return ResNet20(num_classes=10).to(device)

    def instantiate_dataset(self):
        return CIFAR10()

    def instantiate_optimizer(self, digital_lr: float, digital_momentum: float, analog_lr: float, analog_momentum: float):
        digital_parameters, analog_parameters = self.digital_analog_parameters(self.model)
        return AnalogSGD(
            [
                {
                    "params": analog_parameters,
                    "lr": analog_lr,
                    "momentum": analog_momentum,
                },
                {
                    "params": digital_parameters,
                    "lr": digital_lr,
                    "momentum": digital_momentum,
                },
            ],
        )

    def instantiate_scheduler(self):
        assert self.model is not None
        assert self.optimizer is not None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    
    def train(self, num_steps: int, batch_size: int, num_workers: int, logging_freq: int = 50):
        assert self.model is not None
        assert self.optimizer is not None
        train_dataloader = self.dataset.load_train_data(batch_size=batch_size, num_workers=num_workers, validation=False)
        self.model.train().to(device)
        criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        current_step = 0
        while current_step < num_steps:
            for batch_idx, (data, target) in enumerate(train_dataloader):
                data, target = data.to(device), target.to(device)
                for param in self.model.parameters():
                    param.grad = None

                output = self.model(data)
                loss = criterion(output, target)
                if not torch.isnan(loss).any():
                    loss.backward()
                    self.optimizer.step()
                    loss += loss.item() * data.size(0)

                if self.scheduler is not None:
                    self.scheduler.step()

                if (current_step == 0 or (current_step + 1) % logging_freq == 0):
                    logging.info("current_step: %d,\tloss: %2.2f" % (current_step, loss))

                if current_step > num_steps - 2:
                    return loss / current_step

                current_step += 1

    def evaluate(self, batch_size: int, num_workers: int):
        assert self.model is not None
        test_dataloader = self.dataset.load_train_data(batch_size=batch_size, num_workers=num_workers, validation=True)
        self.model.eval().to(device)
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_dataloader):
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        score = 100.0 * correct / total
        logging.info("Score: %2.2f" % score)
        return score
    
class ResNet8CIFAR10(ResNet20CIFAR10):
    def instantiate_model(self):
        return ResNet8(num_classes=10).to(device)