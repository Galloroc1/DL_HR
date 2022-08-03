import torch
import torch.nn as nn
from .core import BaseCNN
from .utils import LAYER_PARAMS_DICT


class ResNet(BaseCNN):
    # conv [output_channel,kernel_size,stride,padding]
    # pool [kernel_size,stride,padding]
    # input_channel out_channel
    # fc layer input channel and out channel
    # todo ：add BasicBlock
    NAME = 'resnet'
    HEAD_LAYER = LAYER_PARAMS_DICT[NAME]['head']
    BASICBLOCK = [18, 34]
    BOTTLENECK = [50, 101, 152]

    def __init__(self, depth: int = 18, pretrain: bool = False, train: bool = False):
        """
        :param depth: you can choose resnet 18 or other
        """
        assert depth in LAYER_PARAMS_DICT[
            self.NAME].keys(), f'not support depth ,we just support {LAYER_PARAMS_DICT[self.NAME].keys()}'
        super(ResNet, self).__init__(depth=depth, pretrain=pretrain, train=train)
        self.layer_params = LAYER_PARAMS_DICT[self.NAME][self.depth]
        self.tail_params = LAYER_PARAMS_DICT[self.NAME]['tail'][self.depth]
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=7)
        self.fc = torch.nn.Linear(self.tail_params["in_channel"], self.tail_params['out_channel'])
        self.build_model()

    def build_model(self):
        # build head
        for k, v in self.HEAD_LAYER.items():
            if k[0] == "c":
                setattr(self, k, torch.nn.Conv2d(in_channels=self.input_channel, out_channels=v[0], kernel_size=v[1],
                                                 stride=v[2], padding=v[3]))
            elif k[0] == 'b':
                setattr(self, k, torch.nn.BatchNorm2d(v))
            elif k[0] == 'r':
                setattr(self, k, torch.nn.ReLU(inplace=True))
            elif k[0] == 'm':
                setattr(self, k, torch.nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2]))
            else:
                raise ValueError(f"not support type,check value that {v}")

        if self.depth in self.BASICBLOCK:
            for k, v in enumerate(self.layer_params):
                setattr(self, 'layer' + str(k + 1), torch.nn.Sequential())
                for index, item in enumerate(v['BasicBlock']):
                    getattr(self, 'layer' + str(k + 1)).add_module(str(index),
                                                                   BasicBlock(item[0], item[1],
                                                                              down_sample=v['down_sample']))
                    v['down_sample'] = False if v['down_sample'] else v['down_sample']
        else:
            for k, v in enumerate(self.layer_params):
                setattr(self, 'layer' + str(k + 1), torch.nn.Sequential())
                for index, item in enumerate(v['BasicBlock']):
                    getattr(self, 'layer' + str(k + 1)).add_module(str(index),
                                                                   Bottleneck(item[0], item[1], item[2],
                                                                              down_sample=v['down_sample'],
                                                                              down_sample_size=v['down_sample_stride']))
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
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=3, stride=1,
                                     padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)

        if down_sample:
            self.downsample = torch.nn.Sequential()
            self.downsample.add_module("0", torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                                            kernel_size=1,
                                                            stride=stride, padding=0))
            self.downsample.add_module("1", torch.nn.BatchNorm2d(output_channel))
        self.is_down_sample = down_sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: forward data,must be tensor
        :return:basic_block forward result
        """
        identity = x
        # if not down_sample,just add x
        identity = self.down_sample_layer(identity) if self.is_down_sample else identity
        # todo : why bn before conv or after conv
        x = self.activation1(self.bn1(self.conv1(x)))
        x = self.activation2(self.bn2(self.conv2(x)) + identity)
        return x


class Bottleneck(nn.Module):
    def __init__(self, input_channel: int = 64, mid_channel: int = 64, output_channel: int = 64,
                 down_sample: bool = False,
                 down_sample_size: int = 1):
        super(Bottleneck, self).__init__()

        # layer1
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel, out_channels=mid_channel, kernel_size=1,
                                     stride=1, padding=0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(mid_channel)

        # layer2
        self.conv2 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3,
                                     stride=down_sample_size,
                                     padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(mid_channel)

        # layer3
        self.conv3 = torch.nn.Conv2d(in_channels=mid_channel, out_channels=output_channel, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        if down_sample:
            self.downsample = torch.nn.Sequential()
            self.downsample.add_module("0", torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                                                            kernel_size=1,
                                                            stride=down_sample_size, padding=0))
            self.downsample.add_module("1", torch.nn.BatchNorm2d(output_channel))
        # down_sample

        self.is_down_sample = down_sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: forward data,must be tensor
        :return:basic_block forward result
        """
        identity = x
        # if not down_sample,just add x
        identity = self.down_sample_layer(identity) if self.is_down_sample else identity
        # todo : why bn before conv or after conv
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.relu(self.bn3(self.conv3(x)) + identity)
        return x
