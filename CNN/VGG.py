import torch
from .utils import LAYER_PARAMS_DICT
from .core import BaseCNN


class VGG(BaseCNN):
    NAME = 'vgg'

    def __init__(self, depth: int = 16) -> None:
        """
        :param depth: depth of vgg net,include:[11,13,16,19]
        """
        assert depth in LAYER_PARAMS_DICT[self.NAME].keys(), 'not support depth'
        super(VGG, self).__init__(depth=depth)
        self.feature_params = LAYER_PARAMS_DICT[self.NAME][self.depth]
        self.features = torch.nn.Sequential()

        self.classifier_params = LAYER_PARAMS_DICT[self.NAME]["fc_params"]
        self.classifier = torch.nn.Sequential()
        self.build_model()

    def build_model(self) -> None:
        """
        build detail model
        :return: None
        """
        last_channel = self.input_channel
        ind = 0
        for k, v in self.feature_params.items():
            if k[0] == 'p':
                self.features.add_module(str(ind), torch.nn.MaxPool2d(kernel_size=2))
                ind = ind + 1
            else:
                for index, i in enumerate(v):
                    self.features.add_module(str(ind), torch.nn.Conv2d(in_channels=last_channel,
                                                                       out_channels=i,
                                                                       kernel_size=3,
                                                                       padding=1))
                    self.features.add_module(str(ind + 1), torch.nn.ReLU(inplace=True))
                    ind = ind + 2
                    last_channel = i

        ind = 0
        for k, v in self.classifier_params.items():
            if k[0] == 'l':
                self.classifier.add_module(str(ind), torch.nn.Linear(v[0], v[1], bias=True))
            elif k[0] == 'd':
                if v: self.classifier.add_module(str(ind), torch.nn.Dropout(p=0, inplace=False))
            elif k[0] == 'a':
                self.classifier.add_module(str(ind), torch.nn.ReLU(inplace=True))
            else:
                TypeError("not support type")
            ind = ind + 1

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
