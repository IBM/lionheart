import torch
from torch import nn
import aihwkit
from aihwkit.nn import AnalogLinear
from .gen_rpu_config import gen_rpu_config

class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        with_relu=True,
        analog=False,
        rpu_config=None,
        name="fc",
        **kwargs
    ):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_relu = with_relu
        self.name = name
        self.bias = bias
        self.kwargs = kwargs
        self.analog = analog
        self.rpu_config = gen_rpu_config() if rpu_config is None else rpu_config
        if self.analog:
            fc = AnalogLinear(
                in_features, out_features, bias=bias, rpu_config=self.rpu_config
            )
        else:
            fc = nn.Linear(in_features, out_features, bias=bias)

        self.layer = fc
        self.relu = nn.ReLU()

    def forward(self, inputs):
        input_shape = inputs.shape
        outputs = self.layer(inputs)
        output_shape = outputs.shape
        MAC_ops = input_shape[-1] * output_shape[-1]
        if self.bias:
            MAC_ops += output_shape[-1]

        if self.with_relu:
            outputs = self.relu(outputs)

        return outputs, MAC_ops