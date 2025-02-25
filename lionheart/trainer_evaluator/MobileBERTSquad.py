import os
import logging
import torch
from aihwkit.optim import AnalogSGD
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult
from .TrainerEvaluator import TrainerEvaluator
from lionheart.models import Model, MobileBERT
from lionheart.datasets import Squad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_id = "csarron/mobilebert-uncased-squad-v1"

class MobileBERTSquad(TrainerEvaluator):
    def instantiate_model(self):
        return MobileBERT(model_id=model_id).to(device)

    def instantiate_dataset(self):
        return Squad(model_id=model_id, max_seq_len=320)

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
        assert self.optimizer is not None
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
    
    def train(self, num_steps: int, batch_size: int, num_workers: int, logging_freq: int = 50):
        assert self.model is not None
        assert self.optimizer is not None
        train_dataloader = self.dataset.load_train_data(batch_size=batch_size, num_workers=num_workers, validation=False)
        self.model = self.model.train().to(device)
        current_step = 0
        while current_step < num_steps:
            for batch_idx, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "start_positions": batch[3],
                    "end_positions": batch[4],
                }
                outputs = self.model(**inputs)
                loss = outputs[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                self.model.zero_grad(set_to_none=True)
                if (current_step == 0 or (current_step + 1) % logging_freq == 0):
                    logging.info("current_step: %d,\tloss: %2.2f" % (current_step, loss.item()))

                if current_step > num_steps - 2:
                    return loss.mean().item() / current_step

                current_step += 1

    def evaluate(self, batch_size: int, num_workers: int):
        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        assert self.model is not None
        test_dataloader = self.dataset.load_train_data(batch_size=batch_size, num_workers=num_workers, validation=True)
        _, examples, features = self.dataset.load_and_cache_examples(
            num_workers=num_workers, evaluate=True, output_examples=True,
        )
        self.model = self.model.eval().to(device)
        all_results = []
        for batch_idx, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }
                feature_indices = batch[3]
                outputs = self.model(**inputs)

            for i, feature_index in enumerate(feature_indices):
                eval_feature = features[feature_index.item()]
                unique_id = int(eval_feature.unique_id)
                output = [to_list(output[i]) for output in outputs.to_tuple()]
                if len(output) >= 5:
                    start_logits = output[0]
                    start_top_index = output[1]
                    end_logits = output[2]
                    end_top_index = output[3]
                    cls_logits = output[4]
                    result = SquadResult(
                        unique_id,
                        start_logits,
                        end_logits,
                        start_top_index=start_top_index,
                        end_top_index=end_top_index,
                        cls_logits=cls_logits,
                    )
                else:
                    start_logits, end_logits = output
                    result = SquadResult(unique_id, start_logits, end_logits)

                all_results.append(result)

        output_prediction_file = os.path.join("predictions.json")
        output_nbest_file = os.path.join("nbest_predictions.json")
        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            20,
            30,
            True,
            output_prediction_file,
            output_nbest_file,
            None,
            False,
            False,
            0.0,
            self.model.tokenizer,
        )
        results = squad_evaluate(examples, predictions)
        score = results['best_f1']
        logging.info("Score: %2.2f" % score)
        return score