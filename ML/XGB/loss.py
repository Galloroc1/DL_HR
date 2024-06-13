import numpy as np
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
