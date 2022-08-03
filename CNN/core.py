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

    def load_state_dict_(self, drop_key_lens: int = None, pretrain: bool = False):
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
