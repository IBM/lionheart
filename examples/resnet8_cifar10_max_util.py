import lionheart
import torch
import logging
import numpy as np
from lionheart.trainer_evaluator.ResNetCIFAR10 import ResNet8CIFAR10
from lionheart.methods.LH import LH, LHConfig
from lionheart.models.common.Linear import Linear
from lionheart.models.common.Conv2d import Conv2d

class LHMaxUtil(LH):
    def det_candidate_order(self):
        self.config.trainer_evaluator.set_model()
        self.config.trainer_evaluator.model.convert_layers_to_analog(None)
        average_tile_util = {}
        for name, module in self.config.trainer_evaluator.model.named_modules():
            if isinstance(module, (Linear, Conv2d)):
                utils = []
                for analog_tile in module.layer.analog_tiles():
                    area = analog_tile.in_size * analog_tile.out_size
                    util = area / (analog_tile.rpu_config.mapping.max_input_size * analog_tile.rpu_config.mapping.max_output_size)
                    assert(util <= 1.0)
                    utils.append(util)

                average_tile_util[name] = np.mean(utils).item()

        self.layer_candidates = list(
            {
                k: v
                for k, v in sorted(
                    average_tile_util.items(), key=lambda item: item[1], reverse=True
                )
            }.keys()
        )
        for candidate in self.layer_candidates:
            self.determine_id(candidate)

        filtered_layer_candidates = []
        for candidate in self.layer_candidates:
            candidate_id = self.determine_id(candidate)
            if candidate_id not in self.ind_analog_layers:
                filtered_layer_candidates.append(candidate_id)

        self.layer_candidates = filtered_layer_candidates

        logging.info("Layer candidate order: %s" % self.layer_candidates)

if __name__ == "__main__":
    config = LHConfig(
        trainer_evaluator=ResNet8CIFAR10(),
        checkpoint_path='resnet8_cifar10.pt',
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
    lh = LHMaxUtil(config)
    lh.set_baseline_score()
    lh.run()