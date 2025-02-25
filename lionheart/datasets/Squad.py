import os
import torch
from transformers import AutoTokenizer, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor
from .Dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

class Squad(Dataset):
    def __init__(self, model_id: str, max_seq_len: int = 320):
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            do_lower_case=True,
            cache_dir=os.path.join(os.getcwd(), 'data'),
            use_fast=False,
        )

    def load_and_cache_examples(
        self,
        num_workers,
        evaluate=True,
        output_examples=False,
        overwrite_cache=False,
    ):
        cached_features_file = os.path.join(
            "cached_{}_{}_{}".format(
                "dev" if evaluate else "train",
                list(filter(None, self.model_id.split("/"))).pop(),
                str(self.max_seq_len),
            ),
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            features_and_dataset = torch.load(cached_features_file)
            features, dataset, examples = (
                features_and_dataset["features"],
                features_and_dataset["dataset"],
                features_and_dataset["examples"],
            )
        else:
            import tensorflow_datasets as tfds
            from tensorflow_datasets.core.utils import gcs_utils
            gcs_utils._is_gcs_disabled = True
            tfds_examples = tfds.load("squad", data_dir=os.path.join(os.getcwd(), 'data'), try_gcs=False)
            examples = SquadV1Processor().get_examples_from_dataset(
                tfds_examples, evaluate=evaluate
            )
            features, dataset = squad_convert_examples_to_features(
                examples=examples,
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_len,
                doc_stride=128,
                max_query_length=64,
                is_training=not evaluate,
                return_dataset="pt",
                threads=num_workers,
            )
            torch.save(
                {"features": features, "dataset": dataset, "examples": examples},
                cached_features_file,
            )

        if output_examples:
            return dataset, examples, features

        return dataset

    def load_train_data(
        self,
        batch_size: int,
        num_workers: int,
        validation: bool,
    ):
        if validation:
            return self.load_test_data(batch_size=batch_size, shuffle=False, num_workers=num_workers)
        else:
            dataset, _, _ = self.load_and_cache_examples(
                num_workers=num_workers, evaluate=False, output_examples=True,
            )
            train_sampler = RandomSampler(dataset)
            return DataLoader(
                dataset, sampler=train_sampler, batch_size=batch_size,
            )

    def load_test_data(
        self,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ):
        assert shuffle == False
        dataset, _, _ = self.load_and_cache_examples(
            num_workers=num_workers, evaluate=True, output_examples=True,
        )
        eval_sampler = SequentialSampler(dataset)
        return DataLoader(
            dataset, sampler=eval_sampler, batch_size=batch_size,
        )