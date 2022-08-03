import torch
from CNN.ResNet import ResNet
import numpy as np
from torchvision.models import resnet50

data = torch.tensor(np.random.random_sample((2, 3, 224, 224)), dtype=torch.float32)
net = ResNet(depth=50)
net.load_state_dict_(drop_key_lens=None)
