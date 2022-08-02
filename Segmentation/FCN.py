import torch
from .core import BaseSegNet
from CNN.VGG import VGG


class FCN(BaseSegNet):

    def __init__(self, num_classes=1000, types=8):
        super(FCN, self).__init__()
        self.num_classes = num_classes
        self.types = types
        self.feature = torch.nn.Sequential()
        self.fc_conv = torch.nn.Sequential()
        self.list_pool_index = None
        self.deconvolution = torch.nn.Sequential()
        self.build_model()

    def build_model(self):
        self.feature = VGG().feature_layer
        dict_layer = self.feature.__dict__['_modules']
        self.list_pool_index = [list(dict_layer.keys()).index(x) + 1 for x in ['pool3', 'pool4', 'pool5']]

        self.fc_conv.add_module("fc_conv1", torch.nn.Conv2d(512, 4096, kernel_size=1, stride=1, padding=0))
        self.fc_conv.add_module("fc_conv2", torch.nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0))
        if self.types == 32:
            self.deconvolution.add_module("de_conv32",
                                          torch.nn.ConvTranspose2d(4096, self.num_classes, kernel_size=32, stride=32))
        elif self.types == 16:
            self.deconvolution.add_module("de_conv16_1",
                                          torch.nn.ConvTranspose2d(4096, 512, kernel_size=2, stride=2))
            self.deconvolution.add_module("de_conv16_2",
                                          torch.nn.ConvTranspose2d(512, self.num_classes, kernel_size=16, stride=16))
        elif self.types == 8:
            self.deconvolution.add_module("de_conv8_1",
                                          torch.nn.ConvTranspose2d(4096, 256, kernel_size=4, stride=4))
            self.deconvolution.add_module("de_conv8_2",
                                          torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))
            self.deconvolution.add_module("de_conv8_3",
                                          torch.nn.ConvTranspose2d(256, self.num_classes, kernel_size=8, stride=8))
        else:
            ValueError(f"not support value: {self.types}")

        self.sequential.add_module("feature", self.feature)
        self.sequential.add_module("fc_conv", self.fc_conv)

    def forward(self, x):
        pool3 = self.feature[0:self.list_pool_index[0]](x)
        pool4 = self.feature[self.list_pool_index[0]:self.list_pool_index[1]](pool3)
        pool5 = self.feature[self.list_pool_index[1]:](pool4)
        fc_conv = self.fc_conv(pool5)

        x = self.deconvolution[0](fc_conv)

        if self.types == 16:
            x = self.deconvolution[1](x+pool4)
        elif self.types == 8:
            pool4_2_de = self.deconvolution[1](pool4)
            x = self.deconvolution[2](x+pool4_2_de)
        elif self.types == 32:
            x = x
        else:
            ValueError(f"not support value: {self.types}")
        return x
