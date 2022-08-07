from abc import ABC
import pandas as pd
import numpy as np
import cv2
import torch
import torchvision
from torch.utils.data.dataset import Dataset
import os


class DataSetBase(Dataset):
    ROOT_PATH = '/home/icaro/pyproject/DL_HR/data/'

    def __init__(self):
        super(DataSetBase, self).__init__()
        self.data_path = None
        self.types = 'base'
        self.name = 'base'

    def read_data(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
