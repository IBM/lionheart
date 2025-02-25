import torch
from torch import nn
import aihwkit
from aihwkit.nn import AnalogConv2d
from .gen_rpu_config import gen_rpu_config

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        k_size,
        stride=1,
        padding=0,
        bias=True,
        dilation=1,
        with_bn=True,
        with_relu=True,
        analog=False,
        rpu_config=None,
        name="conv",
        **kwargs
    ):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.bias = bias
        self.name = name
        self.kwargs = kwargs
        self.name = name
        self.analog = analog
        self.rpu_config = gen_rpu_config() if rpu_config is None else rpu_config
        if self.analog:
            convolution = AnalogConv2d(
                in_channels,
                out_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
                dilation=dilation,
                rpu_config=self.rpu_config,
            )
        else:
            convolution = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                bias=bias,
                dilation=dilation,
            )

        self.layer = convolution
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        outputs = self.layer(inputs)
        output_shape = outputs.shape
        k_size = (self.k_size, self.k_size) if type(self.k_size) == int else self.k_size
        MAC_ops = (
            k_size[0]
            * k_size[1]
            * self.in_channels
            * self.out_channels
            * output_shape[-1]
            * output_shape[-2]
        )
        if self.bias:
            MAC_ops += self.out_channels

        if self.with_bn:
            outputs = self.batchnorm(outputs)

        if self.with_relu:
            outputs = self.relu(outputs)

        return outputs, MAC_ops