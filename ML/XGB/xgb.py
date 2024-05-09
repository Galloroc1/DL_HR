import numpy as np
import sys

from dataclasses import dataclass
from copy import deepcopy


class MseLoss:

    @classmethod
    def g(cls, y_pred, y_true):
        return y_pred - y_true

    @classmethod
    def h(cls, y_pred, y_true):
        return np.ones(shape=y_true.shape)

    @classmethod
    def forward(cls, y_pred: np.ndarray, y_true: np.ndarray):
        return 1 / 2 * (y_true - y_pred) ** 2

    @classmethod
    def g_h(cls, y_pred, y_true):
        return np.array([y_pred - y_true, np.ones(shape=y_true.shape)]).T


@dataclass
class ModelArgs:
    bucket_num = 10
    beta = 0
    alpha = 1
    loss_function = MseLoss
    max_depth = 5
    tree_nums = 10
    task_type = "reg"
    lr = 1
    gamma = 0


def get_split_index(gain):
    max_index_flat = np.argmax(gain)
    max_x_index, max_y_index = np.unravel_index(max_index_flat, gain.shape)
    max_gain = gain[max_x_index, max_y_index]
    return max_gain, max_x_index, max_y_index


def get_max_gain_left_index(buckets, max_x_index, max_y_index):
    left_index = buckets[max_x_index, :, max_y_index]
    right_index = ~left_index
    return left_index, right_index


def get_left_right_node(x, g, h, G_H, max_x_index, max_y_index, other_features):
    # todo : may be better
    ids_left = x[0:max_x_index, max_y_index]
    ids_right = x[max_x_index:, max_y_index]

    left = np.isin(x, ids_left).T
    right = ~left

    x = x.T
    x_left = x[left].reshape((x.shape[0], -1)).T
    x_right = x[right].reshape((x.shape[0], -1)).T

    g = g.T
    h = h.T
    g_left = g[left].reshape((g.shape[0], -1)).T
    g_right = g[right].reshape((g.shape[0], -1)).T

    h_left = h[left].reshape((h.shape[0], -1)).T
    h_right = h[right].reshape((h.shape[0], -1)).T

    G_left = np.sum(g_left, axis=0)[0]
    H_left = np.sum(h_left, axis=0)[0]

    G_right = G_H[0] - G_left
    H_right = G_H[1] - H_left
    return (ids_left, ids_right,
            [x_left[:, other_features], g_left[:, other_features], h_left[:, other_features], [G_left, H_left]],
            [x_right[:, other_features], g_right[:, other_features], h_right[:, other_features], [G_right, H_right]])


def del_feature(feature_lens, max_y_index, org_features):
    other_features = list(range(0, feature_lens))
    other_features.remove(max_y_index)

    split_point_y = org_features[max_y_index]
    org_features.remove(split_point_y)
    return other_features, org_features, split_point_y


class Node:

    def __init__(self, depth, pred_shape, args: ModelArgs):
        self.depth = depth
        self.pred_shape = pred_shape

        self.task_type = args.task_type
        self.loss_function = args.loss_function
        self.bucket_num = args.bucket_num
        self.beta = args.beta
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.args = args

        self.left_node = None
        self.right_node = None

        self.w = None
        self.split_point = None

    def __str__(self):
        params = {
            "w": self.w,
            "depth": self.depth,
            "point": self.split_point
            # "beat": self.beta,
            # "alpha": self.alpha,
            # "loss_function": self.loss_function.__name__
        }
        return str(params)

    def is_leaf(self, x):
        return len(x) < self.bucket_num or self.depth == 0 or x.shape[1] <= 1

    def split_gain(self, x, g, h, G_H):
        def func(bucket, item, g, h, G_H):
            bucket[0:item] = True
            bucket[item:] = False
            G_left = np.sum(g * bucket, axis=0)
            G_right = G_H[0] - G_left
            H_left = np.sum(h * bucket, axis=0)
            H_right = G_H[1] - H_left

            gain_left = G_left ** 2 / (H_left + self.alpha)
            gain_right = G_right ** 2 / (H_right + self.alpha)
            gain_l_r = G_H[0] ** 2 / (G_H[1] + self.alpha)
            gain = 0.5 * (gain_left + gain_right - gain_l_r) - self.beta
            return gain

        bucket_index = [int(x.shape[0] * i / self.bucket_num) for i in range(1, self.bucket_num + 1)][:-1]
        gain = np.array(list(map(lambda item: func(deepcopy(x), item, g, h, G_H), bucket_index)))
        return gain, bucket_index

    def get_w(self, G_H):
        return - G_H[0] / (G_H[1] + self.alpha)

    def train(self, x, g, h, G_H, org_features):
        is_leaf = self.is_leaf(x)
        if is_leaf:
            self.w = self.get_w(G_H)
            new_pred = np.zeros(self.pred_shape)
            new_pred[x[:, 0]] = self.w
            return new_pred
        else:

            gains, buckets_index = self.split_gain(x, g, h, G_H)
            assert gains.shape == (self.bucket_num - 1, x.shape[1]), f"gain shape is wrong:{gains.shape}"

            # find max index and split data
            max_gain, max_gain_x_index, max_gain_y_index = get_split_index(gains)
            max_buckets_index = buckets_index[max_gain_x_index]
            if max_gain > self.gamma:
                self.left_node = Node(self.depth - 1, self.pred_shape, self.args)
                self.right_node = Node(self.depth - 1, self.pred_shape, self.args)
                # del feature
                other_features, org_features, point_y = del_feature(x.shape[-1], max_gain_y_index,
                                                                    org_features)

                # find the split point index in x
                self.split_point = [x[max_buckets_index, max_gain_y_index], point_y]

                # split left ids and right ids
                ids_left, ids_right, left_datas, right_datas = get_left_right_node(x, g, h, G_H,
                                                                                   max_buckets_index,
                                                                                   max_gain_y_index,
                                                                                   other_features)
                left_pred = self.left_node.train(*(left_datas + [deepcopy(org_features)]))
                right_pred = self.right_node.train(*(right_datas + [deepcopy(org_features)]))
                new_pred = left_pred + right_pred
                return new_pred
            else:
                self.w = self.get_w(G_H)
                new_pred = np.zeros(self.pred_shape)
                new_pred[x[:, 0]] = self.w
                return new_pred

    def predict(self, x):
        if self.split_point is not None:
            pred = np.zeros(x.shape[0])
            left = x[:, self.split_point[1]] < self.w
            right = ~left
            pred_left = self.left_node.predict(x[left])
            right_left = self.right_node.predict(x[right])
            pred[left] = pred_left
            pred[right] = right_left
            return pred
        else:
            return self.w

    def get_split_value(self, x):
        if self.split_point is not None:
            self.w = x[self.split_point[0], self.split_point[1]]
            self.left_node.get_split_value(x)
            self.right_node.get_split_value(x)
        else:
            pass


def show_node(node_now):
    if node_now is not None:
        print(node_now)
        show_node(node_now.left_node)
        show_node(node_now.right_node)


class XgboostTree:

    def __init__(self, agrs: ModelArgs):
        self.model_args = agrs
        self.loss_function = self.model_args.loss_function
        self.root_node = None

    def compute_g_h(self, x, y, pred):
        g = self.loss_function.g(pred, y)
        h = self.loss_function.h(pred, y)
        G_H = [np.sum(g, axis=0), np.sum(h, axis=0)]
        assert 0 < len(y.shape) <= 2, f"y.shape has some error,just support 2D or 1D, y.shape ={y.shape}"
        assert len(x.shape) == 2, f"x.shape has some error,just support 2D or 1D, x.shape ={x.shape}"

        if len(y.shape) == 1:
            g = np.expand_dims(g, axis=1)
            h = np.expand_dims(h, axis=1)

        g = np.take_along_axis(g, x, axis=0)
        h = np.take_along_axis(h, x, axis=0)

        return g, h, G_H

    def train(self, x, y, pred):
        g, h, G_H = self.compute_g_h(x, y, pred)
        self.root_node = Node(self.model_args.max_depth, pred.shape, self.model_args)
        feature_list = list(range(x.shape[1]))
        new_pred = self.root_node.train(x, g, h, G_H, feature_list)
        return new_pred

    def predict(self, x):
        return self.root_node.predict(x)

    def get_split_value(self, x):
        self.root_node.get_split_value(x)

    def show(self):
        show_node(self.root_node)


class Xgboost:

    def __init__(self, agrs: ModelArgs):
        self.model_args = agrs
        self.trees = []

    def train(self, x, y):
        x_sort = np.argsort(x, 0)
        pred = np.zeros(y.shape)
        for tree in range(self.model_args.tree_nums):
            model = XgboostTree(self.model_args)
            pred = pred + self.model_args.lr * model.train(x_sort, y, pred)
            loss = MseLoss.forward(pred, y).mean()
            print(loss)
            model.get_split_value(x)
            # show_node(model.root_node)
            self.trees.append(model)
        return pred

    def predict(self, x):
        pred = np.zeros(x.shape[0])
        for tree in self.trees:
            pred = pred + self.model_args.lr * tree.predict(x)
        return pred


if __name__ == '__main__':
    # data = pd.read_csv("../data/motor_hetero_guest.csv")
    # x = data.values[:, 2:]
    # y = data.values[:, 1:2]
    #
    sample = (200000, 50)
    x = np.random.random_sample(sample)
    y = np.random.random_sample((sample[0], 1))

    model = Xgboost(ModelArgs())
    p = model.train(x, y)
    # print(y[0:10])
    # print(p[0:10,0])
    # print("*" * 100)
    # pred = model.predict(x).reshape((-1,1))
    # print(pred[0:10,0])
    # loss = MseLoss.forward(pred, y).mean()
    # print(loss)

    # # sample = (200000,50)
    # # x = np.random.random_sample(sample)
    # # y = np.random.random_sample(sample[0])
    #
    # import matplotlib.pyplot as plt
    #
    # # 准备数据
    # shape = 200
    # x = list(range(0, len(y[0:shape])))
    # # 创建折线图
    # plt.figure(figsize=(12, 6))
    # plt.bar(x, y[0:shape], color="red")
    # plt.plot(x, pred[0:shape], color="blue")
    # # 添加标题和轴标签
    # plt.title("Simple Line Chart")
    # plt.xlabel("x values")
    # plt.ylabel("y values (x squared)")
    #
    # # 显示网格
    # plt.grid(True)
    #
    # # 显示图表
    # plt.show()
    # plt.savefig("test.png")
