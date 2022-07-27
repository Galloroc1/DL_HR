import torch.nn as nn
import logging


class BaseCNN(nn.Module):

    def __init__(self, depth: int) -> None:
        super(BaseCNN, self).__init__()
        self.depth = depth
        self.input_channel = 3
        self.sequential = nn.Sequential()

    def forward(self, x):
        raise

    def build_model(self):
        raise

    def detail(self, is_print=False):
        for i in self.sequential:
            logging.log(1, i)
            print(i) if is_print else None
