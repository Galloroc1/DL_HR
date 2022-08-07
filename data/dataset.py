import numpy as np
from abc import ABC
from data.core import DataSetBase
import os
import pandas as pd
from PIL import Image
from uitl import rle_decode
import torch
import torchvision


class InriaAerial(DataSetBase, ABC):

    def __init__(self, train: bool = True,
                 transform_img=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                 transform_label=None):

        super(InriaAerial, self).__init__()
        self.name = 'InriaAerial'
        self.types = 'Segmentation'
        self.train = train
        self.data_path = os.path.join(self.ROOT_PATH, self.types, self.name)
        self.img_path = os.path.join(self.data_path, 'train' if self.train else 'test')
        self.img = None
        self.label = None
        self.transform_img = transform_img
        self.transform_label = transform_label
        self.read_data()

    def read_data(self):
        file_list = os.listdir(self.img_path)
        assert len(file_list) > 0, 'check your train_set path whether exists image'
        if self.train:
            train_mask = pd.read_csv(os.path.join(self.data_path, 'train_mask.csv'), sep='\t', names=['name', 'mask'])
            train_mask = train_mask.dropna(axis=0).reset_index(inplace=False).drop(columns=['index'])
            self.img = train_mask['name']
            self.label = train_mask['mask']
        else:
            self.img = file_list
        # mask = rle_decode(train_mask['mask'].iloc[0])

    def __getitem__(self, index):
        img = self.transform_img(Image.open(os.path.join(self.img_path, self.img[index])))
        mask = rle_decode(self.label.iloc[index])
        return img, mask

    def __len__(self):
        return len(self.img)


def demo():
    r = InriaAerial()
    train_loader = torch.utils.data.DataLoader(dataset=InriaAerial(),
                                               batch_size=10,
                                               shuffle=False)
    for k, v in train_loader:
        print(k.shape, v.shape)
        break
