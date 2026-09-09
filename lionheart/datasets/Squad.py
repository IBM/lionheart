import os
import torch
from transformers import AutoTokenizer, squad_convert_examples_to_features
from transformers.data.processors.squad import SquadExample, SquadFeatures, SquadV1Processor
from .Dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from lionheart.huggingface import MOBILEBERT_SQUAD_REVISION

class Squad(Dataset):
    CACHE_FORMAT_VERSION = 1

    def __init__(
        self,
        model_id: str,
        max_seq_len: int = 320,
        model_revision: str = MOBILEBERT_SQUAD_REVISION,
    ):
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.model_revision = model_revision
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.model_revision,
            do_lower_case=True,
            cache_dir=os.path.join(os.getcwd(), 'data'),
            use_fast=False,
        )

    @classmethod
    def _to_safe_cache_value(cls, value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "item") and callable(value.item):
            return value.item()
        if hasattr(value, "tolist") and callable(value.tolist):
            return value.tolist()
        if isinstance(value, dict):
            return {
                cls._to_safe_cache_value(key): cls._to_safe_cache_value(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(cls._to_safe_cache_value(item) for item in value)
        if isinstance(value, list):
            return [cls._to_safe_cache_value(item) for item in value]
        return value

    @classmethod
    def _serialize_feature(cls, feature):
        return cls._to_safe_cache_value({
            "input_ids": feature.input_ids,
            "attention_mask": feature.attention_mask,
            "token_type_ids": feature.token_type_ids,
            "cls_index": feature.cls_index,
            "p_mask": feature.p_mask,
            "example_index": feature.example_index,
            "unique_id": feature.unique_id,
            "paragraph_len": feature.paragraph_len,
            "token_is_max_context": feature.token_is_max_context,
            "tokens": feature.tokens,
            "token_to_orig_map": feature.token_to_orig_map,
            "start_position": feature.start_position,
            "end_position": feature.end_position,
            "is_impossible": feature.is_impossible,
            "qas_id": feature.qas_id,
        })

    @staticmethod
    def _deserialize_feature(feature):
        return SquadFeatures(
            input_ids=feature["input_ids"],
            attention_mask=feature["attention_mask"],
            token_type_ids=feature["token_type_ids"],
            cls_index=feature["cls_index"],
            p_mask=feature["p_mask"],
            example_index=feature["example_index"],
            unique_id=feature["unique_id"],
            paragraph_len=feature["paragraph_len"],
            token_is_max_context=feature["token_is_max_context"],
            tokens=feature["tokens"],
            token_to_orig_map=feature["token_to_orig_map"],
            start_position=feature["start_position"],
            end_position=feature["end_position"],
            is_impossible=feature["is_impossible"],
            qas_id=feature["qas_id"],
        )

    @classmethod
    def _serialize_example(cls, example):
        return cls._to_safe_cache_value({
            "qas_id": example.qas_id,
            "question_text": example.question_text,
            "context_text": example.context_text,
            "answer_text": example.answer_text,
            "start_position_character": (
                example.char_to_word_offset.index(example.start_position)
                if (
                    example.answer_text is not None
                    and example.char_to_word_offset
                    and example.start_position in example.char_to_word_offset
                )
                else None
            ),
            "title": example.title,
            "answers": example.answers,
            "is_impossible": example.is_impossible,
        })

    @staticmethod
    def _deserialize_example(example):
        return SquadExample(
            qas_id=example["qas_id"],
            question_text=example["question_text"],
            context_text=example["context_text"],
            answer_text=example["answer_text"],
            start_position_character=example["start_position_character"],
            title=example["title"],
            answers=example["answers"],
            is_impossible=example["is_impossible"],
        )

    @classmethod
    def _serialize_cache(cls, features, dataset, examples):
        return {
            "cache_format_version": cls.CACHE_FORMAT_VERSION,
            "features": [cls._serialize_feature(feature) for feature in features],
            "dataset_tensors": dataset.tensors,
            "examples": [cls._serialize_example(example) for example in examples],
        }

    @classmethod
    def _deserialize_cache(cls, cached):
        if cached.get("cache_format_version") != cls.CACHE_FORMAT_VERSION:
            raise ValueError("Unsupported SQuAD cache format; regenerate the cache.")

        features = [
            cls._deserialize_feature(feature)
            for feature in cached["features"]
        ]
        dataset = TensorDataset(*cached["dataset_tensors"])
        examples = [
            cls._deserialize_example(example)
            for example in cached["examples"]
        ]
        return features, dataset, examples

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
            features, dataset, examples = self._deserialize_cache(
                torch.load(cached_features_file, weights_only=True)
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
                self._serialize_cache(features, dataset, examples),
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
