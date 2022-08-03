import torch
import torch.nn as nn
import logging
from .utils import model_urls
import os


class BaseCNN(nn.Module):
    NAME = "Base"

    def __init__(self, depth: int) -> None:
        super(BaseCNN, self).__init__()
        self.depth = depth
        self.input_channel = 3

    def forward(self, x):
        raise

    def build_model(self):
        raise

    def load_state_dict_(self, drop_key_lens: int = None,
                         fine_tuning: bool = True):
        """
        arg:
            drop_key_lens:int , how many layer you want drop.
                example :  static_key = {'layer1.weight':1,
                                        'layer1.bias':2,
                                        'layer2.weight':1,
                                        'layer2.bias':0}
                            if drop_key_lens = 1, we will drop 'layer2.weight' and 'layer2.bias'
            fine_tuning : bool , is keep base layer's
        """
        from torch.hub import load_state_dict_from_url
        url = model_urls[self.NAME][str(self.depth)]
        model_dir = os.path.join(os.path.abspath(".."), "model_params")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if os.path.isfile(os.path.join(model_dir, url.split("/")[-1])):
            static = torch.load(os.path.join(model_dir, url.split("/")[-1]))
        else:
            static = load_state_dict_from_url(url=url, model_dir=model_dir)
        if drop_key_lens is not None:
            drop_key = list(static.keys())[-drop_key_lens * 2:]
            for i in drop_key:
                static.pop(i)
        self.load_state_dict(state_dict=static, strict=False)
        if fine_tuning:
            for k, v in enumerate(self.parameters()):
                if k < len(static.keys()):
                    v.requires_grad = False
