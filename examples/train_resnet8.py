import lionheart
import torch
import logging
from lionheart.trainer_evaluator.ResNetCIFAR10 import ResNet8CIFAR10

torch.autograd.set_detect_anomaly(True)
train_batch_size = 256
eval_batch_size = 512
num_workers = 1
epochs = 200
te = ResNet8CIFAR10()
te.set_model()
te.set_optimizer(
    digital_lr=1e-2,
    digital_momentum=0.85,
    analog_lr=0.,
    analog_momentum=0.,
)
te.set_scheduler()
num_steps = len(te.dataset.load_train_data(batch_size=train_batch_size, num_workers=num_workers, validation=False))
best_score = 0
for epoch in range(epochs):
    te.train(
        num_steps=num_steps,
        batch_size=train_batch_size, 
        num_workers=num_workers,
        logging_freq=100,
    )
    score = te.evaluate(
        batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    logging.info("Epoch: %d,\tScore: %2.2f" % (epoch, score))
    if score > best_score:
        te.save_checkpoint(checkpoint_path="resnet.pt")
        logging.info("New best score: %2.2f" % (score))
        best_score = score
