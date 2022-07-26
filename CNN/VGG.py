import torch
import torch.nn as nn
from .core import BaseCNN
from uitl import check_params_type_int


class VGG(nn.Module, BaseCNN):
    LAYER_PARAMS = {
        11: {"conv1": [64], "pool1": 'M', "conv2": [128], "pool2": "M", "conv3": [256, 256], "pool3": "M",
             "conv4": [512, 512], "pool4": "M", "conv5": [512, 512], "pool5": "M"},
        13: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256], "pool3": "M",
             "conv4": [512, 512], "pool4": "M", "conv5": [512, 512], "pool5": "M"},
        16: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256, 256], "pool3": "M",
             "conv4": [512, 512, 512], "pool4": "M", "conv5": [512, 512, 512], "pool5": "M"},
        19: {"conv1": [64, 64], "pool1": 'M', "conv2": [128, 128], "pool2": "M", "conv3": [256, 256, 256, 256],
             "pool3": "M", "conv4": [512, 512, 512, 512], "pool4": "M", "conv5": [512, 512, 512, 512],
             "pool5": "M"},
    }

    def __init__(self, depth: int = 16,) -> None:

        nn.Module.__init__(self)
        BaseCNN.__init__(self)
        # params check
        assert depth in self.LAYER_PARAMS.keys(), 'not support depth'
        # init sequential
        self.depth = depth

    def build_model(self) -> None:
        last_channel = self.input_channel
        for k, v in self.LAYER_PARAMS[self.depth].items():
            if k[0:4] == 'pool':
                self.sequential.add_module(k, torch.nn.MaxPool2d(kernel_size=2))
            else:
                for index, i in enumerate(v):
                    self.sequential.add_module(k + str(index), torch.nn.Conv2d(in_channels=last_channel,
                                                                               out_channels=i,
                                                                               kernel_size=3,
                                                                               padding=1))
                    self.sequential.add_module(k + str(index) + "activation", torch.nn.ReLU())
                    last_channel = i

    def forward(self, x):
        r = self.sequential(x)
        return r

