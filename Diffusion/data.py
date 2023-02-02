import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_s_curve
import torch


def get_torch_dataloader(batch_size=124):
    dataset = torch.tensor(load_data(), dtype=torch.float)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_data():
    s_curve, _ = make_s_curve(10 ** 4, noise=0.1)
    s_curve = s_curve[:, [0, 2]] / 10.0
    return s_curve


def show_data(data, title):
    fig, axs = plt.subplots()
    axs.scatter(data[:, 0], data[:, 1], color="red", edgecolors="white")
    # axs[j, k].set_axis_off()
    # axs[j, k].set_title(title)
    axs.axis("off")
