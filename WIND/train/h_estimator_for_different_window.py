import argparse
import os

import torch
import pathlib
import pandas as pd
import numpy as np
from WIND.data_loader import data_loader
from h_estimator import compute_Hc
from statsmodels.tsa.seasonal import seasonal_decompose
from fBM import FractionalBM


def h_estimate_func(noise, kind, min_window, max_window):
    """
        The kind parameter of the compute_Hc function can have the following values:
        'change': a series is just random values (i.e. np.random.randn(...))
        'random_walk': a series is a cumulative sum of changes (i.e. np.cumsum(np.random.randn(...)))
        'price': a series is a cumulative product of changes (i.e. np.cumprod(1+epsilon*np.random.randn(...))
        ==> therefore: kind 'random_walk' is for ABM, 'price' for GBM
    """
    h, _, _ = compute_Hc(noise, kind=kind, min_window=min_window, max_window=max_window, simplified=False)
    return h

def h_over_residuals():

    file_names = ['res_additive','res_multiplicative', 'logres_additive']

    tmp_results = []
    for file_name in file_names:
        file = pathlib.Path(f'../data/{file_name}.csv')
        df = pd.read_csv(file, index_col=0, header=0, sep=',')
        noise = df.resid.values
        min_w_range = np.arange(20, 365, 10)
        max_w_range = np.arange(365, len(noise) - 1, 31)
        for min_w in min_w_range:
            for max_w in max_w_range:
                h = h_estimate_func(noise, 'change', min_w, max_w)
                tmp_results.append((min_w, max_w,h))
        results_df = pd.DataFrame(data=tmp_results, columns=['min_w', 'max_w', 'h_estimated'])
        results_df.to_csv(f"{file_name}-h.csv", index=False)


def h_over_fractional(h_index, min_w, max_w):
    mock_params = [1, 0.00, 0.05, 0.10, h_index]  # (S0, mu, theta, sigma, H)
    fBM = FractionalBM()
    fBM.set_parameters(mock_params)
    sims_df = fBM.simulate(n_sims=1, t_steps=365 * 5, dt=1 / 365)
    noise = np.squeeze(sims_df.values)
    h = h_estimate_func(noise, 'price', min_window=min_w, max_window=max_w)
    return max(min(h, 1), 0)


if __name__ == '__main__':
    h_vec = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
    n_sims = 1000
    min_w = np.arange(30, 365 * 4, 60)
    max_w = np.arange(60, 365 * 5 - 1, 60)
    df_h = pd.DataFrame(data=[], columns=max_w, index=min_w)
    df_vol = pd.DataFrame(data=[], columns=max_w, index=min_w)
    for h in h_vec:
        for min_idx in min_w:
            for max_idx in max_w:
                h_estimated = []
                dev = 0
                for n in range(0, n_sims):
                    h_tmp = h_over_fractional(h, min_idx, max_idx)
                    h_estimated.append(h_tmp)
                    dev += (h_tmp - h) ** 2
                    df_h.loc[min_idx, max_idx] = h_estimated
                    print(f'w_min: {min_idx}\tw_max:{max_idx}\tH: Estimated: {h_tmp}\t Expected: {h}')
                df_vol.loc[min_idx, max_idx] = np.sqrt(dev / n_sims)
        os.makedirs("results", exist_ok=True)
        df_vol.to_csv(f'results/H_{h}-vol.csv', index=True)
        df_h.to_csv(f'results/H_{h}-dist.csv', index=True)






