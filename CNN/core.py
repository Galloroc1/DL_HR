import torch.nn as nn
import logging
class BaseCNN():

    def __init__(self) -> None:
        self.sequential = nn.Sequential()
        self.input_channel = 3


    def build_model(self):
        pass

    def detail(self,is_print=False):
        for i in self.sequential:
            logging.log(1,i)
            print(i) if is_print else None
