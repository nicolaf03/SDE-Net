import pathlib
from hurst import compute_Hc
import pandas as pd
import numpy as np


def h_estimate_func():
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

    np.random.seed(42)
    noise = np.random.randn(1000)

    # Alternative versions tested.
    # noise = np.cumsum(x * 0.10 * np.sqrt(4/1000) + 0.01 * 4/1000)  # dX_t = 0 * dt + 0.1 * dW_t
    # noise = np.cumprod(1 + x * 0.10 * np.sqrt(4 / 1000) + 0.01 * 4/1000)  # dX_t / X_t = 0 * dt + 0.1 * dW_t
    h, _, _ = compute_Hc(noise, kind='change', min_window=20, max_window=365, simplified=True)
    return h


def get_noise(prices: np.array, kind: str):
    match kind:
        case 'gbm':
            r = np.diff(np.log(prices))
        case 'abm':
            r = np.diff(prices)
        case _:
            r = np.diff(prices)
    return (r - r.mean()) / r.std()


if __name__ == '__main__':
    h = h_estimate_func()
    print(h)





