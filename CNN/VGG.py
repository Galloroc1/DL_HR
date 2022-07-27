import torch
from .utils import LAYER_PARAMS_DICT
from .core import BaseCNN


class VGG(BaseCNN):
    NAME = 'vgg'

    def __init__(self, depth: int = 16, ) -> None:
        """
        :param depth: depth of vgg net,include:[11,13,16,19]
        """
        assert depth in LAYER_PARAMS_DICT[self.NAME].keys(), 'not support depth'
        super(VGG, self).__init__(depth=depth)
        self.feature_params = LAYER_PARAMS_DICT[self.NAME][self.depth]
        self.feature_layer = torch.nn.Sequential()

        self.classifier_params = LAYER_PARAMS_DICT[self.NAME]["fc_params"]
        self.classifier_layer = torch.nn.Sequential()
        self.build_model()

    def build_model(self) -> None:
        """
        build detail model
        :return: None
        """
        last_channel = self.input_channel
        for k, v in self.feature_params.items():
            if k[0] == 'p':
                self.feature_layer.add_module(k, torch.nn.MaxPool2d(kernel_size=2))
            else:
                for index, i in enumerate(v):
                    self.feature_layer.add_module(k + "_" + str(index + 1), torch.nn.Conv2d(in_channels=last_channel,
                                                                                            out_channels=i,
                                                                                            kernel_size=3,
                                                                                            padding=1))
                    self.feature_layer.add_module(k + "_" + str(index + 1) + "_activation", torch.nn.ReLU(inplace=True))
                    last_channel = i
        for k, v in self.classifier_params.items():
            if k[0] == 'l':
                self.classifier_layer.add_module(k, torch.nn.Linear(v[0], v[1], bias=True))
            elif k[0] == 'd':
                if v: self.classifier_layer.add_module(k, torch.nn.Dropout(p=0.5, inplace=False))
            elif k[0] == 'a':
                self.classifier_layer.add_module(k, torch.nn.ReLU(inplace=True))
            else:
                TypeError("not support type")
        self.sequential.add_module("feature", self.feature_layer)
        self.sequential.add_module("class", self.classifier_layer)

    def forward(self, x):
        x = self.feature_layer(x)
        x = torch.flatten(x, 1)
        x = self.classifier_layer(x)
        return x
