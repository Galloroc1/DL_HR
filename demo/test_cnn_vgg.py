# todo add root path
import torch.cuda
from tqdm import tqdm
from CNN.VGG import VGG
import torchvision

"""
    in this demo ,you should change ./utils.py LAYER_PARAMS_DICT.vgg.fc_params,
    line 64 : [4096,10] , 10 is mean 10 class_num
"""
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize(224)])
train_set = torchvision.datasets.CIFAR10(root="./data", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, transform=dataset_transform, download=True)
batch_size = 32
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                          batch_size=1,
                                          shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGG().cuda()
model.load_state_dict_(drop_key_lens=3, fine_tuning=True)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())
loss_image = []
for data in tqdm(train_loader):
    score = model.forward(data[0].cuda())
    loss = loss_func(score, data[1].cuda())
    loss.backward()
    loss_image.append(loss.cpu().detach().numpy())
    optimizer.step()
    optimizer.zero_grad()
    # pred = torch.argmax(score, dim=1)

# from matplotlib import pyplot as plt
# plt.figure()
# plt.plot(range(len(loss_image)),loss_image,color='red')
# plt.show()

prt = 0
total = 0
for data in tqdm(test_loader):
    test_input, test_label = data
    test_output = model.forward(test_input.cuda())
    total += 1
    if torch.argmax(test_output, dim=1) == test_label.cuda():
        prt += 1
print("accuracy=\t", prt / total)
