import torch
import logging


class BaseSegNet(torch.nn.Module):

    def __init__(self):
        super(BaseSegNet, self).__init__()
        self.sequential = torch.nn.Sequential()

    def forward(self, x):
        raise

    def build_model(self):
        raise
