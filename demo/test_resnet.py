import torch
from CNN.ResNet import ResNet
import numpy as np

data = torch.tensor(np.random.random_sample((2,3,224,224)),dtype=torch.float32)
net = ResNet(depth=34)
net.detail(is_print=True)
r = net.forward(data)
