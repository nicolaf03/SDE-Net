import math
import numpy as np


def rmse(y_pred, y_true):
    err = y_true - y_pred
    return math.sqrt((err * err).mean())


def mape_lgb(y_true, y_pred):
    return np.mean( np.abs(y_true - y_pred) / np.maximum(1.0, np.abs(y_true)) )


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mape_func(y_true, y_pred):
    return np.abs((y_true - y_pred) / y_true)



