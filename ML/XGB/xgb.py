import numpy as np
import sys

sys.path.append("../../fl")
from dataclasses import dataclass
from copy import deepcopy
from tools import timer_decorator

np.set_printoptions(precision=15)


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
    max_depth = 6
    tree_nums = 100
    task_type = "reg"
    lr = 1
    gamma = 0


# @timer_decorator
def get_split_index(gain):
    max_index_flat = np.argmax(gain)
    max_x_index, max_y_index = np.unravel_index(max_index_flat, gain.shape)
    max_gain = gain[max_x_index, max_y_index]
    return max_gain, max_x_index, max_y_index


def get_max_gain_left_index(buckets, max_x_index, max_y_index):
    left_index = buckets[max_x_index, :, max_y_index]
    right_index = ~left_index
    return left_index, right_index


def get_part(x, part):
    x = x.T[part].reshape((x.shape[1], -1)).T
    return x


# @timer_decorator
def get_left_right_node(x, mask, g, h, G_H, max_x_index, max_y_index, other_features):
    # todo : may be better
    left_ids_mask = mask[max_x_index, :, max_y_index]
    right_ids_mask = ~left_ids_mask

    ids_left = x[..., max_y_index, 0][left_ids_mask]
    ids_right = x[..., max_y_index, 0][right_ids_mask]
    mask_left = np.isin(x[..., 0], ids_left).T
    mask_right = ~mask_left

    x0 = np.expand_dims(get_part(x[..., 0], mask_left), -1)
    x1 = np.expand_dims(get_part(x[..., 1], mask_left), -1)
    x_left = np.concatenate((x0, x1), -1)

    x0 = np.expand_dims(get_part(x[..., 0], mask_right), -1)
    x1 = np.expand_dims(get_part(x[..., 1], mask_right), -1)
    x_right = np.concatenate((x0, x1), -1)

    g_left = get_part(g, mask_left)
    g_right = get_part(g, mask_right)
    G_left = np.sum(g_left, axis=0)[0]
    G_right = G_H[0] - G_left

    h_left = get_part(h, mask_left)
    h_right = get_part(h, mask_right)
    H_left = np.sum(h_left, axis=0)[0]
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
        }
        return str(params)

    def is_leaf(self, x):
        return self.depth == 0 or x.shape[1] <= 1 or x.shape[0] <= 1

    # @timer_decorator
    def split_gain(self, x, g, h, G_H):

        def func1(bucket, item):
            result = np.empty(bucket.shape[0:2], dtype=bool)
            for i in range(0, bucket.shape[1]):
                result[:, i] = bucket[:, i, 1] < bucket[item, i, 1]
            return result

        def func2(bucket, g, h, G_H):
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
        mask = np.array(list(map(lambda item: func1(deepcopy(x), item), bucket_index)))
        gain = np.array(list(map(lambda item: func2(item, g, h, G_H), mask)))
        return gain, bucket_index, mask

    def get_w(self, G_H):
        return - G_H[0] / (G_H[1] + self.alpha)

    def train(self, x, g, h, G_H, org_features):
        is_leaf = self.is_leaf(x)

        if is_leaf:
            self.w = self.get_w(G_H)
            new_pred = np.zeros(self.pred_shape)
            new_pred[x[:, 0, 0]] = self.w
            return new_pred
        else:
            gains, buckets_index, mask = self.split_gain(x, g, h, G_H)
            assert gains.shape == (self.bucket_num - 1, x.shape[1]), f"gain shape is wrong:{gains.shape}"
            # find max index and split data
            max_gain, max_gain_x_index, max_gain_y_index = get_split_index(gains)

            if max_gain > self.gamma:
                self.left_node = Node(self.depth - 1, self.pred_shape, self.args)
                self.right_node = Node(self.depth - 1, self.pred_shape, self.args)
                other_features, org_features, point_y = del_feature(x.shape[-2],
                                                                    max_gain_y_index,
                                                                    org_features)
                self.split_point = [x[buckets_index[max_gain_x_index], max_gain_y_index, 0], point_y]

                ids_left, ids_right, left_datas, right_datas = get_left_right_node(x, mask, g, h, G_H,
                                                                                   max_gain_x_index,
                                                                                   max_gain_y_index,
                                                                                   other_features)
                left_pred = self.left_node.train(*(left_datas + [deepcopy(org_features)]))
                right_pred = self.right_node.train(*(right_datas + [deepcopy(org_features)]))
                new_pred = left_pred + right_pred
                return new_pred
            else:
                self.w = self.get_w(G_H)
                new_pred = np.zeros(self.pred_shape)
                new_pred[x[:, 0, 0]] = self.w
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
        G_H = [np.sum(g, axis=0)[0], np.sum(h, axis=0)[0]]
        assert 0 < len(y.shape) <= 2, f"y.shape has some error,just support 2D or 1D, y.shape ={y.shape}"
        assert len(x.shape) == 3, f"x.shape has some error,just support 2D or 1D, x.shape ={x.shape}"

        if len(y.shape) == 1:
            g = np.expand_dims(g, axis=1)
            h = np.expand_dims(h, axis=1)

        g = np.take_along_axis(g, x[..., 0], axis=0)

        h = np.take_along_axis(h, x[..., 0], axis=0)

        return g, h, G_H

    @timer_decorator
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

    @timer_decorator
    def rebuild_x(self, x):
        x_sort = np.expand_dims(np.argsort(x, 0), -1)
        result = np.empty(shape=x.shape + (1,))
        for col in range(x.shape[1]):
            unique_vals, inverse = np.unique(x[:, col], return_inverse=True)
            result[:, col, 0] = inverse
        eq_matrix = np.take_along_axis(result, x_sort, 0)
        x_sort = np.concatenate([x_sort, eq_matrix], axis=-1).astype(int)
        return x_sort

    @timer_decorator
    def train(self, x, y):
        x_sort = self.rebuild_x(x)
        pred = np.zeros(y.shape)
        for tree in range(self.model_args.tree_nums):
            model = XgboostTree(self.model_args)
            new_pred = self.model_args.lr * model.train(x_sort, y, pred)
            pred = pred + new_pred
            loss = MseLoss.forward(pred, y).mean()
            print(loss)
            model.get_split_value(x)
            self.trees.append(model)
        return pred

    def predict(self, x):
        pred = np.zeros(x.shape[0])
        for tree in self.trees:
            new_pred = self.model_args.lr * tree.predict(x)
            pred = pred + new_pred
        return pred


if __name__ == '__main__':
    import numpy as np
    import pandas as pd

    data = pd.read_csv("../data/motor_hetero_guest.csv")
    x = data.values[:, 2:]
    y = data.values[:, 1:2]

    # size = 200000
    # x = np.random.random_sample((size,50))
    # y = np.random.random_sample((size,1))

    args = ModelArgs()
    model = Xgboost(args)
    p = model.train(x, y)
    pred = model.predict(x).reshape((-1, 1))
    loss = MseLoss.forward(pred, y).mean()
    print("predictloss", loss)
    # print("pred 0", pred[0])
    index = pred != p
    print(sum(index))
    import matplotlib.pyplot as plt

    # 准备数据
    shape = 200
    x = list(range(0, len(y[0:shape])))
    # 创建折线图
    plt.figure(figsize=(12, 6))
    plt.bar(x, y[0:shape].flatten(), color="red")
    plt.plot(x, pred[0:shape].flatten(), color="blue")
    # 添加标题和轴标签
    plt.title("Simple Line Chart")
    plt.xlabel("x values")
    plt.ylabel("y values (x squared)")

    # 显示网格
    plt.grid(True)

    # 显示图表
    plt.show()
    plt.savefig("test.png")
