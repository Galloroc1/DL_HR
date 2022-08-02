from .core import BaseSegNet
import torch


class UNet(BaseSegNet):

    def __init__(self):
        super(UNet, self).__init__()

    def build_model(self):
        pass


class ConvBlock(torch.nn.Module):

    def __init__(self, in_channel: int = 64, out_channel: int = 64, mid_channel: int = 64):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channel = mid_channel

        self.conv_stride = 1
        self.kernel_size = 3
        self.padding = 1

        # not origin u-net origin:padding = 0
        self.conv_1 = torch.nn.Conv2d(self.in_channel, self.mid_channel, self.kernel_size, self.conv_stride,
                                      self.padding)
        self.relu_1 = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(self.in_channel, self.mid_channel, self.kernel_size, self.conv_stride,
                                      self.padding)
        self.relu_2 = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        return x
