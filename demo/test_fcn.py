from Segmentation.FCN import FCN
import torch
import numpy as np

data = torch.tensor(np.random.random_sample((1, 3, 224, 224)), dtype=torch.float)
net = FCN(num_classes=10)
net.load_state_dict_(depth=16)
x = net.forward(data)
print(x.shape)
