from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import lionheart
from aihwkit.nn.conversion import convert_to_digital
from aihwkit.nn import AnalogConv2d, AnalogLinear
from aihwkit.inference.utils import drift_analog_weights as aihwkit_drift_analog_weights
from .common.Linear import Linear
from .common.Conv2d import Conv2d
from lionheart.methods.utils import rgetattr, rsetattr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Model(ABC, nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.MAC_ops = {}
        self.track_MACs = True

    @abstractmethod
    def forward(self):
        pass

    def convert_layers_to_analog(self, ind_analog_layers: list[int]):
        self.convert_layers_to_digital()
        self.replacement_layers = []
        ind = 0
        for i, (name, module) in enumerate(self.named_modules()):
            analog = ind in ind_analog_layers
            convert = False
            try:
                if isinstance(rgetattr(self, ".".join(name.split(".")[0:-1])), (Linear, Conv2d)):
                    continue
            except:
                pass

            if isinstance(module, nn.Linear):
                new_module = Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=True if module.bias is not None else False,
                    with_relu=False,
                    analog=analog,
                    name=name,
                )
                convert = True
            elif isinstance(module, nn.Conv2d):
                new_module = Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    k_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    bias=True if module.bias is not None else False,
                    dilation=module.dilation,
                    with_bn=False,
                    with_relu=False,
                    analog=analog,
                    name=name,
                )
                convert = True

            if convert:
                weight = module.weight
                bias = module.bias
                new_module.ind_analog_layer = ind
                ind += 1
                if analog:
                    new_module.layer.set_weights(weight, bias)
                else:
                    new_module.layer.weight = weight
                    new_module.layer.bias = bias

                rsetattr(self, name, new_module)
                self.replacement_layers.append(name)

        for i, (name, module) in enumerate(self.named_modules()):
            if name in self.replacement_layers:
                module.register_forward_hook(self.layer_hook(name))

    def layer_hook(self, layer_name):
        def hook(module, input, output):
            if self.track_MACs:
                if layer_name not in self.MAC_ops:
                    self.MAC_ops[layer_name] = 0

                self.MAC_ops[layer_name] += output[1]

            return output[0]

        return hook
    
    def convert_layers_to_digital(self):
        for i, (name, module) in enumerate(self.named_modules()):
            try:
                if isinstance(rgetattr(self, ".".join(name.split(".")[0:-1])), (Linear, Conv2d)):
                    continue
            except:
                pass

            if isinstance(module, Linear) or isinstance(module, Conv2d):
                if module.analog:
                    layer = convert_to_digital(module.layer)
                else:
                    layer = module.layer

                rsetattr(self, name, layer)

    def drift_analog_weights(self, t_inference: float):
        for i, (name, module) in enumerate(self.named_modules()):
            if ".".join(name.split(".")[0:-1]) == "":
                continue

            if isinstance(module, Linear) or isinstance(module, Conv2d):
                aihwkit_drift_analog_weights(module.layer, t_inference=t_inference)

    def set_track_MACs(self):
        self.track_MACs = True

    def unset_track_MACs(self):
        self.track_MACs = False

    def get_MACs(self):
        return self.MAC_ops
