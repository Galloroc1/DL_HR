import numpy as np
import sys
from dataclasses import dataclass
from copy import deepcopy


class MseLoss:

    @classmethod
    def g(cls, y_pred, y_true):
        return y_true - y_pred

    @classmethod
    def h(cls, y, t):
        return 1

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


def get_left_right_index_matrix(sort_args, bucket_index, ):
    def func(arr, item):
        arr[0:item] = True
        arr[item:] = False
        return arr

    left = map(lambda item: func(deepcopy(sort_args), item), bucket_index)
    return np.array(list(left))


def get_split_index(gain):
    max_index_flat = np.argmax(gain)
    max_x_index, max_y_index = np.unravel_index(max_index_flat, gain.shape)
    return max_x_index, max_y_index


def get_max_gain_left_index(buckets, max_x_index, max_y_index):
    left_index = buckets[max_x_index, :, max_y_index]
    right_index = ~left_index
    return left_index, right_index


def get_left_right_node(x, g_h, G_H, max_x_index, max_y_index, other_features):
    # todo : may be better
    ids_left = x[0:max_x_index, max_y_index]
    ids_right = x[max_x_index:, max_y_index]

    left = np.isin(x, ids_left)
    right = ~left

    x_left = x.T[left.T].reshape((x.shape[1], -1)).T
    x_right = x.T[right.T].reshape((x.shape[1], -1)).T

    g_h_left = np.dstack([g_h[:, :, i].T[left.T].reshape((g_h.shape[1], -1)).T for i in range(g_h.shape[-1])])
    g_h_right = np.dstack([g_h[:, :, i].T[right.T].reshape((g_h.shape[1], -1)).T for i in range(g_h.shape[-1])])

    G_H_left = np.sum(g_h_left, axis=0)[0]
    G_H_right = G_H - G_H_left
    return (ids_left, ids_right, [x_left[:, other_features], g_h_left[:, other_features], G_H_left],
            [x_right[:, other_features], g_h_right[:, other_features], G_H_right])


def del_feature(feature_lens, max_y_index, features_list):
    other_features = list(range(0, feature_lens))
    other_features.remove(max_y_index)

    split_point_y = features_list[max_y_index]
    features_list.remove(split_point_y)
    return other_features, features_list, split_point_y


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

    def get_bucket_index(self, x):
        split_point = [int(x.shape[0] * i / self.bucket_num) for i in range(1, self.bucket_num + 1)][:-1]
        return split_point

    # 1
    def compute_bucket_g(self, bucket, g_h, G_H):
        bucket = np.expand_dims(bucket, axis=-1) * g_h
        G_H_left = np.sum(bucket, axis=1)
        G_H_right = G_H - G_H_left
        gain_left = G_H_left[:, :, 0] ** 2 / (G_H_left[:, :, 1] + self.alpha)
        gain_right = G_H_right[:, :, 0] ** 2 / (G_H_right[:, :, 1] + self.alpha)
        gain_l_r = G_H[0] ** 2 / (G_H[1] + self.alpha)
        gain = 0.5 * (gain_left + gain_right - gain_l_r) - self.beta
        return gain

    # 1
    def split_bucket(self, x):
        bucket_index = self.get_bucket_index(x)
        left_index = get_left_right_index_matrix(x, bucket_index)
        assert left_index.shape[0] == self.bucket_num - 1, f"split has wrong"
        return left_index, bucket_index

    def get_w(self, G_H=None):
        if G_H is None and self.w is None:
            raise "some error"

        if G_H is not None:
            w = - G_H[0] / (G_H[1] + self.alpha)
            return w
        else:
            return self.w

    def train(self, x, g_h=None, G_H=None, features_list=None):
        is_leaf = self.is_leaf(x)
        if is_leaf:
            self.w = self.get_w(G_H)
            new_pred = np.zeros(self.pred_shape)
            new_pred[x[:, 0]] = self.w
            return new_pred
        else:
            buckets, buckets_index = self.split_bucket(x)
            gains = self.compute_bucket_g(buckets, g_h, G_H)
            assert gains.shape == (self.bucket_num - 1, x.shape[1]), f"gain shape is wrong"

            # find max index and split data
            max_x_index, max_y_index = get_split_index(gains)
            max_gain = gains[max_x_index, max_y_index]

            if max_gain > self.gamma:
                self.left_node = Node(self.depth - 1, self.pred_shape, self.args)
                self.right_node = Node(self.depth - 1, self.pred_shape, self.args)
                # del feature
                other_features, features_list, split_point_y = del_feature(buckets.shape[-1], max_y_index,
                                                                           features_list)

                self.split_point = [x[buckets_index[max_x_index], max_y_index], split_point_y]

                ids_left, ids_right, left_datas, right_datas = get_left_right_node(x, g_h, G_H,
                                                                                   buckets_index[max_x_index],
                                                                                   max_y_index, other_features)
                left_pred = self.left_node.train(*(left_datas + [deepcopy(features_list)]))
                right_pred = self.right_node.train(*(right_datas + [deepcopy(features_list)]))
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
        g_h = self.loss_function.g_h(pred, y)
        new_x = np.expand_dims(x, axis=-1)
        new_x = np.tile(new_x, (1, 1, 2))

        G_H = np.sum(g_h, axis=0)
        g_h = np.expand_dims(g_h, axis=1)
        g_h = np.tile(g_h, (1, x.shape[1], 1))
        g_h = np.take_along_axis(g_h, new_x, axis=0)
        return g_h, G_H

    def train(self, x, y, pred):
        g_h, G_H = self.compute_g_h(x, y, pred)
        self.root_node = Node(self.model_args.max_depth, pred.shape[0], self.model_args)
        feature_list = list(range(x.shape[1]))
        new_pred = self.root_node.train(x, g_h, G_H, feature_list)
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
        x_sort = np_sort(x)
        pred = np.zeros(len(y))
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
        pred = np.zeros(len(x))
        for tree in self.trees:
            pred = pred + self.model_args.lr * tree.predict(x)
        return pred


def np_sort(x):
    sort_arg = np.argsort(x, 0)
    return sort_arg


if __name__ == '__main__':
    # data = pd.read_csv("../data/motor_hetero_guest.csv")
    # x = data.values[:, 2:]
    # y = data.values[:, 1]

    sample = (200000, 50)
    x = np.random.random_sample(sample)
    y = np.random.random_sample(sample[0])

    model = Xgboost(ModelArgs())
    p = model.train(x, y)
    # print(y[0:10])
    # print(p[0:10])
    # print("*" * 100)
    # pred = model.predict(x)
    # print(pred[0:10])
    #
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
