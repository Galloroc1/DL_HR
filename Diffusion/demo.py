from PIL import Image
import numpy as np


def get_data(path="1.png"):
    img = Image.open(path)
    matrix = np.array(img)
    return matrix[:, :, 0:3]


def add_noise(shape):
    step = 100
    beta = 1 - np.sort(np.random.random_sample(step))
    return np.random.uniform(0, 1, tuple([step] + list(shape))) * beta.reshape((-1, 1, 1, 1)), beta


def diffusion(x, n, b):
    new = []
    for i in range(n.shape[0]):
        x = np.sqrt(b[i]) * x + np.sqrt(1 - b[i]) * n[i]
        # x = b[i] * x + (1 - b[i]) * n[i]
        new.append(x)
    return x, new


def show_ndarray(k, x):
    x = np.uint8(x * 255)
    img = Image.fromarray(x)
    img.show()


def de_diffusion(x, b, n, new):
    b = b[::-1]
    n = n[::-1]
    for i in range(0, len(n)):
        print(new[-(i + 1)][0, 0, :] - x[0, 0, :])
        x = (x - np.sqrt(1 - b[i]) * n[i]) / np.sqrt(b[i])

    return x


matrix = get_data()
noise, bata = add_noise(matrix.shape)
d_matrix, new_ = diffusion(matrix, noise, bata)
for k, v in enumerate(new_):
    show_ndarray(k, v)
org_matrix = de_diffusion(d_matrix, bata, noise, new_)
show_ndarray(org_matrix)
