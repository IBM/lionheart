import os
import uuid
import logging
import torch
import copy
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from aihwkit.nn import AnalogConv2d, AnalogLinear
from lionheart.trainer_evaluator import TrainerEvaluator
from .utils import rgetattr, seed_everything

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@dataclass
class LHConfig:
    trainer_evaluator: TrainerEvaluator
    checkpoint_path: str
    train_batch_size: int
    eval_batch_size: int
    digital_lr: float
    digital_momentum: float
    analog_lr: float
    analog_momentum: float
    drop_threshold: float
    num_workers: float = 4
    num_steps: int = -1
    patience: int = 1
    t_eval: float = 86400.0
    evaluation_reps: int = 10
    logging_freq: int = 50
    seed: int = None

class LH():
    def __init__(self, config: LHConfig):
        self.config = config
        self.baseline_score = -1
        self.ind_analog_layers = []
        self.layer_candidates = []
        self.current_layer_candidate = -1
        self.layer_MACs = {}
        self.reverse_id_lookup = {}
        seed_everything(self.config.seed)

    def set_baseline_score(self):
        self.config.trainer_evaluator.set_model()
        self.config.trainer_evaluator.load_checkpoint(self.config.checkpoint_path)
        self.config.trainer_evaluator.model.convert_layers_to_digital()
        self.baseline_score = self.config.trainer_evaluator.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        logging.info("Set baseline score to: %s" % self.baseline_score)

    def determine_id(
        self, candidate,
    ):
        instance = rgetattr(self.config.trainer_evaluator.model, candidate)
        if hasattr(instance, "ind_analog_layer"):
            self.reverse_id_lookup[instance.ind_analog_layer] = candidate
            return instance.ind_analog_layer
        else:
            return -1
        
    def det_analog_digital_mac_ratio(self, checkpoint_path: str, ind_analog_layers: list[int]):
        assert checkpoint_path is not None
        self.config.trainer_evaluator.set_model()
        self.config.trainer_evaluator.load_checkpoint(self.config.checkpoint_path, ind_analog_layers=ind_analog_layers)
        self.config.trainer_evaluator.model.convert_layers_to_analog(ind_analog_layers)
        self.config.trainer_evaluator.model.set_track_MACs()
        self.config.trainer_evaluator.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        MAC_ops = self.config.trainer_evaluator.model.get_MACs()
        analog_MACs = 0
        digital_MACs = 0
        for name, ops in MAC_ops.items():
            layer = rgetattr(self.config.trainer_evaluator.model, name).layer
            if isinstance(layer, (AnalogConv2d, AnalogLinear)):
                analog_MACs += ops
            elif isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                digital_MACs += ops
            else:
                raise NotImplementedError

        self.config.trainer_evaluator.model.unset_track_MACs()
        return {"digital": digital_MACs, "analog": analog_MACs}

    def det_candidate_order(self):
        self.config.trainer_evaluator.set_model()
        self.config.trainer_evaluator.model.convert_layers_to_analog([])
        self.config.trainer_evaluator.model.set_track_MACs()
        self.config.trainer_evaluator.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers)
        MAC_ops = self.config.trainer_evaluator.model.get_MACs()
        self.config.trainer_evaluator.model.unset_track_MACs()
        self.layer_MACs = {
            k: v
            for k, v in sorted(MAC_ops.items(), key=lambda item: item[1], reverse=True)
        }
        self.layer_candidates = list(
            {
                k: v
                for k, v in sorted(
                    MAC_ops.items(), key=lambda item: item[1], reverse=True
                )
            }.keys()
        )
        filtered_layer_candidates = []
        for candidate in self.layer_candidates:
            candidate_id = self.determine_id(candidate)
            if candidate_id not in self.ind_analog_layers:
                filtered_layer_candidates.append(candidate_id)

        self.layer_candidates = filtered_layer_candidates
        logging.info("Layer candidate order: %s" % self.layer_candidates)

    def run(self):
        if self.baseline_score == -1:
            self.set_baseline_score()

        self.uuid = str(uuid.uuid4())
        logging.info("UUID: %s" % self.uuid)
        self.updated_checkpoint_path = os.path.splitext(self.config.checkpoint_path)[0] + "_LH_%s.pt" % (
            self.uuid
        )
        self.ind_analog_layers = self.config.trainer_evaluator.load_checkpoint(checkpoint_path=self.config.checkpoint_path, ind_analog_layers=self.ind_analog_layers)
        if self.ind_analog_layers is None:
            self.ind_analog_layers = []
        
        self.config.trainer_evaluator.save_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)
        self.det_candidate_order()
        if self.config.num_steps == -1:
            self.config.num_steps = len(self.config.trainer_evaluator.dataset.load_train_data(batch_size=self.config.train_batch_size, num_workers=self.config.num_workers, validation=False))
        
        previous_ind_analog_layers = None
        while len(self.layer_candidates) > 0:
            self.current_layer_candidate = self.layer_candidates.pop(0)
            candidate = self.reverse_id_lookup[self.current_layer_candidate]
            logging.info("Selected candidate: %s [%d]" % (candidate, self.current_layer_candidate))
            previous_ind_analog_layers = copy.deepcopy(self.ind_analog_layers)
            self.ind_analog_layers.append(self.current_layer_candidate)
            self.config.trainer_evaluator.model.convert_layers_to_analog(self.ind_analog_layers)
            self.config.trainer_evaluator.set_optimizer(
                digital_lr=self.config.digital_lr,
                digital_momentum=self.config.digital_momentum,
                analog_lr=self.config.analog_lr,
                analog_momentum=self.config.analog_momentum,
            )
            self.config.trainer_evaluator.set_scheduler()
            best_score = 0
            score_queue = deque([best_score] * self.config.patience, maxlen=self.config.patience)
            current_step = 0
            training_converged = False
            while not training_converged:
                self.config.trainer_evaluator.train(
                    num_steps=self.config.num_steps,
                    batch_size=self.config.train_batch_size, 
                    num_workers=self.config.num_workers,
                    logging_freq=self.config.logging_freq,
                )
                current_step += self.config.num_steps
                scores = []
                self.config.trainer_evaluator.model.eval()
                for i in range(self.config.evaluation_reps):
                    self.config.trainer_evaluator.model.drift_analog_weights(t_inference=self.config.t_eval)
                    scores.append(
                        self.config.trainer_evaluator.evaluate(
                            batch_size=self.config.eval_batch_size,
                            num_workers=self.config.num_workers,
                        )
                    )

                avg_score = np.array(scores).mean()
                score_queue.appendleft(avg_score)
                logging.info("Score queue: %s" % score_queue)
                if max(score_queue) < best_score:
                    training_converged = True
                    logging.info("Training converged.")

                if avg_score > best_score:
                    best_score = avg_score
                    self.config.trainer_evaluator.save_checkpoint(checkpoint_path=self.updated_checkpoint_path + ".candidate", ind_analog_layers=self.ind_analog_layers)                    

            if best_score >= (self.baseline_score - self.config.drop_threshold):
                logging.info("Succesfully converted candidate %s [%d]" % (candidate, self.current_layer_candidate))
                os.replace(self.updated_checkpoint_path + ".candidate", self.updated_checkpoint_path)
            else:
                logging.info("Failed to convert candidate: %s [%d]" % (candidate, self.current_layer_candidate))
                os.remove(self.updated_checkpoint_path + ".candidate")
                self.ind_analog_layers = previous_ind_analog_layers

            logging.info("Layer candidate order: %s" % self.layer_candidates)
            self.config.trainer_evaluator.load_checkpoint(checkpoint_path=self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)

        self.config.trainer_evaluator.load_checkpoint(checkpoint_path=self.updated_checkpoint_path)
        logging.info("Checkpoint path: %s" % self.updated_checkpoint_path)
        analog_digital_macs = self.det_analog_digital_mac_ratio(self.updated_checkpoint_path, ind_analog_layers=self.ind_analog_layers)
        logging.info("Analog digital MACs: %s" % analog_digital_macs)
        final_scores = []
        self.config.trainer_evaluator.model.eval()
        for i in range(self.config.evaluation_reps):
            self.config.trainer_evaluator.model.drift_analog_weights(t_inference=self.config.t_eval)
            final_scores.append(self.config.trainer_evaluator.evaluate(batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers))

        final_scores = np.array(final_scores)
        logging.info("Final score: mean %2.2f,\tstd: %2.2f" % (final_scores.mean(), final_scores.std()))
        