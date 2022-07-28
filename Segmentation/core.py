import torch
import torchvision
import logging


class BaseSegNet(torch.nn.Module):

    def __init__(self):
        super(BaseSegNet, self).__init__()
        self.sequential = torch.nn.Sequential()

    def forward(self, x):
        raise

    def build_model(self):
        raise

    def detail(self, is_print=False):
        for i in self.sequential:
            logging.log(1, i)
            print(i) if is_print else None

