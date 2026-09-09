import sys
import types

import torch
from torch.utils.data import TensorDataset
from transformers.data.processors.squad import SquadExample, SquadFeatures


class _ConfigSection:
    pass


class _InferenceRPUConfig:
    def __init__(self):
        self.mapping = _ConfigSection()
        self.remap = _ConfigSection()
        self.clip = _ConfigSection()
        self.modifier = _ConfigSection()


class _EnumValue:
    CHANNELWISE_SYMMETRIC = "channelwise_symmetric"
    FIXED_VALUE = "fixed_value"
    MULT_NORMAL = "mult_normal"


aihwkit = types.ModuleType("aihwkit")
aihwkit_nn = types.ModuleType("aihwkit.nn")
aihwkit_nn.AnalogConv2d = torch.nn.Conv2d
aihwkit_nn.AnalogLinear = torch.nn.Linear
aihwkit_nn_conversion = types.ModuleType("aihwkit.nn.conversion")
aihwkit_nn_conversion.convert_to_digital = lambda module: module
aihwkit_optim = types.ModuleType("aihwkit.optim")
aihwkit_optim.AnalogSGD = torch.optim.SGD
aihwkit_inference = types.ModuleType("aihwkit.inference")
aihwkit_inference.PCMLikeNoiseModel = lambda *args, **kwargs: object()
aihwkit_inference.GlobalDriftCompensation = lambda *args, **kwargs: object()
aihwkit_inference_utils = types.ModuleType("aihwkit.inference.utils")
aihwkit_inference_utils.drift_analog_weights = lambda *args, **kwargs: None
aihwkit_simulator = types.ModuleType("aihwkit.simulator")
aihwkit_simulator_configs = types.ModuleType("aihwkit.simulator.configs")
aihwkit_simulator_configs.InferenceRPUConfig = _InferenceRPUConfig
aihwkit_simulator_configs_utils = types.ModuleType("aihwkit.simulator.configs.utils")
aihwkit_simulator_configs_utils.WeightRemapType = _EnumValue
aihwkit_simulator_configs_utils.WeightModifierType = _EnumValue
aihwkit_simulator_configs_utils.WeightClipType = _EnumValue
aihwkit_simulator_presets = types.ModuleType("aihwkit.simulator.presets")
aihwkit_simulator_presets_utils = types.ModuleType("aihwkit.simulator.presets.utils")
aihwkit_simulator_presets_utils.IOParameters = _ConfigSection
sys.modules.setdefault("aihwkit", aihwkit)
sys.modules.setdefault("aihwkit.nn", aihwkit_nn)
sys.modules.setdefault("aihwkit.nn.conversion", aihwkit_nn_conversion)
sys.modules.setdefault("aihwkit.optim", aihwkit_optim)
sys.modules.setdefault("aihwkit.inference", aihwkit_inference)
sys.modules.setdefault("aihwkit.inference.utils", aihwkit_inference_utils)
sys.modules.setdefault("aihwkit.simulator", aihwkit_simulator)
sys.modules.setdefault("aihwkit.simulator.configs", aihwkit_simulator_configs)
sys.modules.setdefault("aihwkit.simulator.configs.utils", aihwkit_simulator_configs_utils)
sys.modules.setdefault("aihwkit.simulator.presets", aihwkit_simulator_presets)
sys.modules.setdefault("aihwkit.simulator.presets.utils", aihwkit_simulator_presets_utils)

from lionheart.datasets.Squad import Squad
from lionheart.trainer_evaluator.TrainerEvaluator import TrainerEvaluator


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def forward(self, inputs):
        return self.layer(inputs)

    def convert_layers_to_digital(self):
        pass

    def convert_layers_to_analog(self, ind_analog_layers):
        self.ind_analog_layers = ind_analog_layers


class TinyTrainerEvaluator(TrainerEvaluator):
    def instantiate_model(self):
        return TinyModel()

    def instantiate_dataset(self):
        return None

    def instantiate_optimizer(
        self,
        digital_lr: float,
        digital_momentum: float,
        analog_lr: float,
        analog_momentum: float,
    ):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=digital_lr,
            momentum=digital_momentum,
        )

    def instantiate_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)

    def train(self, num_steps: int, batch_size: int, num_workers: int, logging_freq: int):
        pass

    def evaluate(self, batch_size: int, num_workers: int):
        pass


def test_checkpoint_roundtrip_uses_weights_only(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer = TinyTrainerEvaluator()
    trainer.set_model()
    trainer.set_optimizer(
        digital_lr=0.1,
        digital_momentum=0.9,
        analog_lr=0.1,
        analog_momentum=0.9,
    )
    trainer.set_scheduler()

    inputs = torch.ones(1, 2)
    trainer.model(inputs).sum().backward()
    trainer.optimizer.step()
    trainer.scheduler.step()
    expected = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.state_dict().items()
    }
    expected_scheduler = trainer.scheduler.state_dict()

    trainer.save_checkpoint(str(checkpoint_path), ind_analog_layers=[1])
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["checkpoint_format_version"] == TrainerEvaluator.CHECKPOINT_FORMAT_VERSION

    restored = TinyTrainerEvaluator()
    restored.set_model()
    restored.set_optimizer(
        digital_lr=0.1,
        digital_momentum=0.9,
        analog_lr=0.1,
        analog_momentum=0.9,
    )
    restored.set_scheduler()

    with torch.no_grad():
        for parameter in restored.model.parameters():
            parameter.fill_(42.0)

    assert restored.load_checkpoint(str(checkpoint_path)) == [1]
    for name, parameter in restored.model.state_dict().items():
        assert torch.equal(parameter, expected[name])
    assert restored.optimizer.state_dict()["state"]
    assert restored.scheduler.state_dict() == expected_scheduler


def test_squad_cache_roundtrip_uses_weights_only(tmp_path):
    feature = SquadFeatures(
        input_ids=[101, 102],
        attention_mask=[1, 1],
        token_type_ids=[0, 0],
        cls_index=0,
        p_mask=[0.0, 1.0],
        example_index=0,
        unique_id=1000000000,
        paragraph_len=2,
        token_is_max_context={0: True},
        tokens=["[CLS]", "[SEP]"],
        token_to_orig_map={0: 0},
        start_position=0,
        end_position=0,
        is_impossible=False,
        qas_id="question-1",
    )
    dataset = TensorDataset(
        torch.tensor([[101, 102]], dtype=torch.long),
        torch.tensor([[1, 1]], dtype=torch.long),
        torch.tensor([[0, 0]], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([[0.0, 1.0]], dtype=torch.float),
    )
    example = SquadExample(
        qas_id="question-1",
        question_text="Which token?",
        context_text="answer token",
        answer_text="answer",
        start_position_character=0,
        title="title",
        answers=[{"text": "answer", "answer_start": 0}],
        is_impossible=False,
    )
    cache_path = tmp_path / "cached_squad.pt"

    torch.save(Squad._serialize_cache([feature], dataset, [example]), cache_path)
    cached = torch.load(cache_path, weights_only=True)
    features, restored_dataset, examples = Squad._deserialize_cache(cached)

    assert features[0].unique_id == feature.unique_id
    assert features[0].tokens == feature.tokens
    assert torch.equal(restored_dataset.tensors[0], dataset.tensors[0])
    assert examples[0].qas_id == example.qas_id
    assert examples[0].answers == example.answers
