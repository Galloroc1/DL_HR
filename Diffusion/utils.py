import matplotlib.pyplot as plt
import torch


def show_img(x_seq, title):
    fig, axs = plt.subplots(1, 10, figsize=(28, 3))
    for i in range(1, 11):
        cur_x = x_seq[i * 10].cpu().detach()
        axs[i - 1].scatter(cur_x[:, 0], cur_x[:, 1], color='red', edgecolor='white');
        axs[i - 1].set_axis_off()
        axs[i - 1].set_title('$q(\mathbf{x}_{' + str(i * 10) + '})$')
    plt.savefig(f'./img/{str(title)}.png')

# from data import load_data
#
# data = load_data()
# plt.figure()
# plt.scatter(data[:,0],data[:,1],color='red')
# plt.show()
