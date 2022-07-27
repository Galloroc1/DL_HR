import torch
import torchvision
from .core import BaseSegNet
from CNN.VGG import VGG


class FCN(BaseSegNet):

    def __init__(self):
        super(FCN, self).__init__()
        self.feature = torch.nn.Sequential()
        self.deconvolution = torch.nn.Sequential()
        self.build_model()

    def build_model(self):
        self.feature = VGG().feature_layer
        # torch.nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8)
