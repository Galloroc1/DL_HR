import torch
import torch.nn as nn
from .core import BaseCNN


class ResNet(nn.Module, BaseCNN):
    # conv [output_channel,kernel_size,stride,padding]
    # pool [kernel_size,stride,padding]
    HEAD_LAYER = {"conv1": [64, 7, 2, 3], "pool1": [3, 2, 1]}
    # input_channel out_channel
    LAYER_PARAMS = {18: [{"BasicBlock": [[64, 64], [64, 64]],'downsample': False},
                         {"BasicBlock": [[64, 128], [128, 128]],'downsample': True},
                         {"BasicBlock": [[128, 256], [256, 256]],'downsample': True},
                         {"BasicBlock": [[256, 512], [512, 512]],'downsample': True},
                         ]}
    # fc layer input channel and out channel
    FC_PARAMS = {"in_channel": 512, "out_channel": 1000}
    # todo ：add BasicBlock
    BASICBLOCK = [18, 34]

    def __init__(self, depth: int = 18):
        """
        :param depth: you can choose resnet 18 or other
        """
        nn.Module.__init__(self)
        BaseCNN.__init__(self)
        assert depth in self.LAYER_PARAMS.keys(),f'not support depth ,we just support {self.LAYER_PARAMS.keys()}'
        self.depth = depth
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7)
        self.fc = torch.nn.Linear(self.FC_PARAMS["in_channel"],self.FC_PARAMS['out_channel'])

    def build_model(self):
        # build head
        for k, v in self.HEAD_LAYER.items():
            if k[0:4] == "conv":
                self.sequential.add_module(k, torch.nn.Conv2d(in_channels=self.input_channel, out_channels=v[0],
                                                              kernel_size=v[1], stride=v[2], padding=v[3]))
            else:
                self.sequential.add_module(k, torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2]))

        if self.depth in self.BASICBLOCK:
            for k, v in enumerate(self.LAYER_PARAMS[self.depth]):
                for index,item in enumerate(v['BasicBlock']):
                    self.sequential.add_module("BasicBlock_Group" + str(k) + str(index),
                                               BasicBlock(item[0], item[1],
                                                          downsample=v['downsample']))
                    v['downsample'] = False if v['downsample'] else v['downsample']

    def forward(self, x):
        x = self.sequential(x)
        x = self.avg_pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


class BasicBlock(nn.Module):
    """
    basicblock
    """
    def __init__(self, input_channel: int = 64, output_channel: int = 64, downsample: bool = False):
        super(BasicBlock, self).__init__()
        stride = 2 if downsample else 1
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=3,
                                     stride=stride, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.activation1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                     padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.activation2 = torch.nn.ReLU()
        self.downsample_layer = torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=1,
                                                stride=2, padding=0)
        self.downsample = downsample

    def forward(self, x:torch.Tensor)->torch.Tensor:
        """
        :param x: forward data,must be tensor
        :return:basic_block forward result
        """
        identity = x
        # if not downsample,just add x
        identity = self.downsample_layer(identity) if self.downsample else identity
        # todo : why bn before conv or after conv
        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)) + identity)
        return x


class Bottleneck(nn.Module, BaseCNN):
    def __init__(self):
        Bottleneck.__init__(self)
        BaseCNN.__init__(self)
