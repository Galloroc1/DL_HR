from Segmentation.FCN import FCN
import torch
import numpy as np
from torchsummary import summary
from data.dataset import InriaAerial

net = FCN(num_classes=2).cuda()
net.load_state_dict_(depth=16)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=0.01, params=net.parameters())
data_set = InriaAerial()
data_batch = torch.utils.data.DataLoader(dataset=InriaAerial(),
                                         batch_size=10,
                                         shuffle=False)
epoch = 10

for i in range(epoch):
    for x, y in data_batch:
        score = net.forward(x.cuda())
        score = torch.nn.functional.softmax(score, dim=1)
        loss = loss_func(score, y.to(torch.long).cuda())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
