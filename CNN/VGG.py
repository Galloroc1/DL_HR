import torch
from .core import BaseCNN, LAYER_PARAMS_DICT


class VGG(BaseCNN):
    NAME = 'vgg'

    def __init__(self, depth: int = 16, ) -> None:
        assert depth in LAYER_PARAMS_DICT[self.NAME].keys(), 'not support depth'
        super(VGG, self).__init__(depth=depth)
        self.layer_params = LAYER_PARAMS_DICT[self.NAME][self.depth]

    def build_model(self) -> None:
        last_channel = self.input_channel
        for k, v in self.layer_params.items():
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
