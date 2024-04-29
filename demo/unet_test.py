import os
import sys

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from Segmentation.UNet import Unet
import torch.optim as optim

def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, save_path=None):
    """
    Train model
    :param train_loader:
    :param val_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :param save_path:
    """
    best_val_loss = np.inf
    best_model_state = None
    for epoch in tqdm(range(num_epochs)):
        model.train()  # 设置模型为训练模式
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)  # 增加一个维度以匹配模型输入的形状
            print(f"outputs{outputs.shape}")
            inputs('d')
            loss = criterion(outputs, labels)  # 增加一个维度以匹配标签的形状
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 验证模型
        model.eval()    # 将模型设置为评估模式
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # 计算平均损失
        running_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss}, Val Loss: {val_loss}')
        # 保存最佳模型状态
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    print('Finished Training')
    if save_path is not None:
        print(os.path.dirname(save_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
    return best_model_state


if __name__ == '__main__':
    # 虚构的训练数据和标签
    train_data = torch.randn(100, 3, 256, 256)  # 100个尺寸为(3, 256, 256)的图像
    train_labels = torch.randint(0, 2, (100, 256, 256))  # 100个尺寸为(256, 256)的随机标签
    val_data = torch.randn(50, 3, 256, 256)  # 验证集
    val_labels = torch.randint(0, 2, (50, 256, 256))

    test_data = torch.randn(3, 3, 256, 256)  # 验证集
    test_labels = torch.randint(0, 2, (3, 256, 256))

    # 打包并按指定批次组合到一起
    train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_labels)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

    # 创建SegNet模型实例
    model = Unet(class_num=3, in_channels=3)  # 根据实际情况设置输入通道数和类别数

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 调用训练函数
    # model_save_path = os.path.join(sys.path[0], 'model', 'best_model.pt')
    best_model_state = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=2, save_path=None)
    print(f"best_model_state{best_model_state.keys()}")
