import pathlib
from hurst import compute_Hc
import pandas as pd
import numpy as np


def h_estimate_func(series):
    """
        The kind parameter of the compute_Hc function can have the following values:
        'change': a series is just random values (i.e. np.random.randn(...))
        'random_walk': a series is a cumulative sum of changes (i.e. np.cumsum(np.random.randn(...)))
        'price': a series is a cumulative product of changes (i.e. np.cumprod(1+epsilon*np.random.randn(...))
        ==> therefore: kind 'random_walk' is for ABM, 'price' for GBM
    """
    # file = pathlib.Path('../../SDE-Net_1/WIND/data/logres_additive.csv')
    # df = pd.read_csv(file, index_col=0, header=0, sep=',')
    # noise = df.resid.values

    # Alternative versions tested.
    # noise = np.cumsum(x * 0.10 * np.sqrt(4/1000) + 0.01 * 4/1000)  # dX_t = 0 * dt + 0.1 * dW_t
    # noise = np.cumprod(1 + x * 0.10 * np.sqrt(4 / 1000) + 0.01 * 4/1000)  # dX_t / X_t = 0 * dt + 0.1 * dW_t
    h, _, _ = compute_Hc(series, kind='change', min_window=10, max_window=len(series), simplified=True)
    return h


def get_noise(prices: np.array, kind: str):
    if kind == 'gbm':
        r = np.diff(np.log(prices))
    elif kind == 'abm':
        r = np.diff(prices)
    else:
        r = np.diff(prices)
    return (r - r.mean()) / r.std()


if __name__ == '__main__':
    # df = pd.read_csv(file, index_col=0, header=0, sep=',')
    # series = df.resid.values
    file_name_root = r'..\data\res_{}.csv'
    zones = {'CNOR', 'CSUD', 'NORD', 'SARD', 'SICI', 'SUD'}
    n_trials = 100
    for z in zones:
        file = file_name_root.format(z)
        df = pd.read_csv(file, index_col=0, header=0, sep=',')
        series = df.energy.values
        for n in range(0, n_trials):
            h = h_estimate_func(series)
            print('Trial #: {}. Hurst Estimation for {} file: {}'.format(n,file, h))





