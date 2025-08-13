import lionheart
import torch
from lionheart.trainer_evaluator.ResNet20CIFAR10 import ResNet20CIFAR10
from lionheart.methods.LH import LH, LHConfig

if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=ResNet20CIFAR10(),
        checkpoint_path='resnet20_cifar10.pt',
        train_batch_size=128,
        eval_batch_size=256,
        digital_lr=1e-2,
        digital_momentum=0.85,
        analog_lr=1e-3,
        analog_momentum=0.85,
        drop_threshold=2.,
        t_eval=86400.0,
        evaluation_reps=10,
        patience=5,
    )
    lh = LH(config)
    lh.set_baseline_score()
    lh.run()