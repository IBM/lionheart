import pytest
import lionheart
from lionheart.trainer_evaluator.ResNet20CIFAR10 import ResNet20CIFAR10

def test_inference():
    te = ResNet20CIFAR10()
    te.set_model()
    score = te.evaluate(
        batch_size=16,
        num_workers=1,
    )
