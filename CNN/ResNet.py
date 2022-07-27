import torch
import torch.nn as nn
from .core import BaseCNN
from .utils import LAYER_PARAMS_DICT


class ResNet(BaseCNN):
    # conv [output_channel,kernel_size,stride,padding]
    # pool [kernel_size,stride,padding]
    HEAD_LAYER = {"conv1": [64, 7, 2, 3], "pool1": [3, 2, 1]}
    # input_channel out_channel
    # fc layer input channel and out channel
    # todo ：add BasicBlock
    FC_PARAMS = {18: {"in_channel": 512, "out_channel": 1000}, 34: {"in_channel": 512, "out_channel": 1000},
                 50: {"in_channel": 2048, "out_channel": 1000}, 101: {"in_channel": 2048, "out_channel": 1000},
                 152: {"in_channel": 2048, "out_channel": 1000}, }
    BASICBLOCK = [18, 34]
    BOTTLENECK = [50,101,152]
    NAME = 'resnet'

    def __init__(self, depth: int = 18):
        """
        :param depth: you can choose resnet 18 or other
        """
        assert depth in LAYER_PARAMS_DICT[
            self.NAME].keys(), f'not support depth ,we just support {LAYER_PARAMS_DICT[self.NAME].keys()}'
        super(ResNet, self).__init__(depth=depth)
        self.layer_params = LAYER_PARAMS_DICT[self.NAME][self.depth]
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7)
        self.fc = torch.nn.Linear(self.FC_PARAMS[depth]["in_channel"], self.FC_PARAMS[depth]['out_channel'])
        self.build_model()

    def build_model(self):
        # build head
        for k, v in self.HEAD_LAYER.items():
            if k[0:4] == "conv":
                self.sequential.add_module(k, torch.nn.Conv2d(in_channels=self.input_channel, out_channels=v[0],
                                                              kernel_size=v[1], stride=v[2], padding=v[3]))
            else:
                self.sequential.add_module(k, torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2]))

        if self.depth in self.BASICBLOCK:
            for k, v in enumerate(self.layer_params):
                for index, item in enumerate(v['BasicBlock']):
                    self.sequential.add_module("BasicBlock_Group" + str(k) + str(index),
                                               BasicBlock(item[0], item[1],
                                                          down_sample=v['down_sample']))
                    v['down_sample'] = False if v['down_sample'] else v['down_sample']
        else:
            for k, v in enumerate(self.layer_params):
                for index, item in enumerate(v['BasicBlock']):
                    self.sequential.add_module("BasicBlock_Group" + str(k) + str(index),
                                               Bottleneck(item[0], item[1],item[2],
                                                          down_sample=v['down_sample'],down_sample_size=v['down_sample_stride']))
                    v['down_sample'] = False if v['down_sample'] else v['down_sample']
                    v['down_sample_stride'] = 1 if v['down_sample'] else v['down_sample_stride']

    def forward(self, x):
        x = self.sequential(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """
    basicblock
    """

    def __init__(self, input_channel: int = 64, output_channel: int = 64, down_sample: bool = False):
        super(BasicBlock, self).__init__()
        stride = 2 if down_sample else 1
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3,
                                     stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.activation1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                     padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.activation2 = torch.nn.ReLU()
        if down_sample:
            self.down_sample_layer = torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1,
                                                     stride=stride, padding=0)
        self.down_sample = down_sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: forward data,must be tensor
        :return:basic_block forward result
        """
        identity = x
        # if not down_sample,just add x
        identity = self.down_sample_layer(identity) if self.down_sample else identity
        # todo : why bn before conv or after conv
        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)) + identity)
        return x


class Bottleneck(nn.Module):
    def __init__(self, input_channel: int = 64,mid_channel:int = 64 ,output_channel: int = 64, down_sample: bool = False,
                 down_sample_size:int = 1):
        super(Bottleneck, self).__init__()
        if down_sample:
            self.down_sample_layer = torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1,
                                                     stride=down_sample_size, padding=0)
        # layer1
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel, out_channels=mid_channel, kernel_size=1,
                                     stride=1, padding=0,bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_channel)

        # layer2
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride= down_sample_size,
                                     padding=1,bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_channel)

        # layer3
        self.conv3 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=output_channel, kernel_size=1, stride=1,
                                     padding=0,bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.activation = torch.nn.ReLU()

        # down_sample

        self.down_sample = down_sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: forward data,must be tensor
        :return:basic_block forward result
        """
        identity = x
        # if not down_sample,just add x
        identity = self.down_sample_layer(identity) if self.down_sample else identity
        # todo : why bn before conv or after conv
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.activation(self.bn3(self.conv3(x)) + identity)
        return x

