import torch
import torch.nn as nn
from transformers import MobileBertForQuestionAnswering, MobileBertTokenizer
from aihwkit.nn import AnalogLinear
from aihwkit.inference.utils import drift_analog_weights as aihwkit_drift_analog_weights
import lionheart
from .common.Linear import Linear
from .Model import Model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MobileBERT(Model):
    def __init__(self, model_id: str = "csarron/mobilebert-uncased-squad-v1"):
        super(MobileBERT, self).__init__()
        self.model = MobileBertForQuestionAnswering.from_pretrained(model_id)
        self.tokenizer = MobileBertTokenizer.from_pretrained(model_id)

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def convert_layers_to_analog(self, ind_analog_layers: list[int]):
        self.convert_layers_to_digital()
        self.replacement_layers = []
        ind = 0
        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, nn.Linear) and 'embedding' not in name:
                analog = ind in ind_analog_layers
                weight = module.weight
                bias = module.bias
                new_module = Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=True if bias is not None else False,
                    with_relu=False,
                    analog=analog,
                    name=name,
                )
                new_module.ind_analog_layer = ind
                ind += 1
                if analog:
                    new_module.layer.set_weights(weight, bias)
                else:
                    new_module.layer.weight = weight
                    new_module.layer.bias = bias

                lionheart.methods.rsetattr(self.model, name, new_module)
                self.replacement_layers.append(name)

        for i, (name, module) in enumerate(self.model.named_modules()):
            if name in self.replacement_layers:
                module.register_forward_hook(self.layer_hook(name))

    def layer_hook(self, layer_name):
        def hook(module, input, output):
            if self.track_MACs:
                if 'model.' + layer_name not in self.MAC_ops:
                    self.MAC_ops['model.' + layer_name] = 0

                self.MAC_ops['model.' + layer_name] += output[1]

            return output[0]

        return hook
    
    def drift_analog_weights(self, t_inference: float):
        for i, (name, module) in enumerate(self.model.named_modules()):
            if isinstance(module, Linear):
                aihwkit_drift_analog_weights(module.layer, t_inference=t_inference)