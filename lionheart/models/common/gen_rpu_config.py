import aihwkit
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.simulator.configs.utils import (
    WeightRemapType,
    WeightModifierType,
    WeightClipType,
)
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation
from aihwkit.simulator.presets.utils import IOParameters


def gen_rpu_config():
    my_rpu_config = InferenceRPUConfig()
    my_rpu_config.mapping.digital_bias = True
    my_rpu_config.mapping.weight_scaling_omega = 1.0
    my_rpu_config.mapping.weight_scaling_columnwise = True
    my_rpu_config.mapping.learn_out_scaling = True
    my_rpu_config.mapping.out_scaling_columnwise = True
    my_rpu_config.mapping.max_input_size = 256
    my_rpu_config.mapping.max_output_size = 255
    my_rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    my_rpu_config.remap.type = WeightRemapType.CHANNELWISE_SYMMETRIC
    my_rpu_config.clip.type = WeightClipType.FIXED_VALUE
    my_rpu_config.clip.fixed_value = 1.0
    my_rpu_config.modifier.type = WeightModifierType.MULT_NORMAL
    my_rpu_config.modifier.rel_to_actual_wmax = True
    my_rpu_config.modifier.std_dev = 0.08
    my_rpu_config.forward = IOParameters()
    my_rpu_config.forward.out_noise = 0.06
    my_rpu_config.forward.inp_res = 1 / (2**8 - 2)  # 8-bit resolution.
    my_rpu_config.forward.out_res = 1 / (2**8 - 2)  # 8-bit resolution.
    my_rpu_config.drift_compensation = GlobalDriftCompensation()
    return my_rpu_config

