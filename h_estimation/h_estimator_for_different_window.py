import pandas as pd
import numpy as np
import logging

from pathlib import Path
import os
from h_estimator import compute_Hc
from fractional_BM.fBM import FractionalBM

from multiprocessing import Pool

# module logger
logger = logging.getLogger(__name__)

curr_dir = Path(__file__).parent


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


def generate_fbm_trajectories(h_index, n_sims, T:int ):
    mock_params = [1, 0.00, 0.05, 0.10, h_index]  # (S0, mu, theta, sigma, H)
    fBM = FractionalBM()
    fBM.set_parameters(mock_params)
    sims_df = fBM.simulate(n_sims=n_sims, t_steps=T, dt=1 / 365)
    noise = np.squeeze(sims_df.values)
    return noise


if __name__ == '__main__':
    # INPUT
    h_vec = [.2, .3, .4, .5, .6, .7, .8, .9]
    n_sims = 1000
    freq = 365
    t_n = freq * 5
    step = 30
    min_windows = np.arange(30, t_n - 365, step)
    max_windows = np.arange(100, t_n - 1, step)
    
    def your_function(h):
        print(f'Try with h = {h}')
        
        df_h = pd.DataFrame(data=[], columns=max_windows, index=min_windows)
        df_rmse = pd.DataFrame(data=[], columns=max_windows, index=min_windows)
        noise = generate_fbm_trajectories(h, n_sims, t_n)
        logger.info(f'{n_sims} for fbm with H = {h} generated.')
        for min_w in min_windows:
            for max_w in max_windows:
                h_estimated = []
                rmse = 0
                for n in range(0, n_sims):
                    h_tmp = h_estimate_func(noise[n, :], 'price', min_window=min_w, max_window=max_w)
                    h_estimated.append(h_tmp)
                    rmse += (h_tmp - h) ** 2
                    logger.info(f'w_min: {min_w}\tw_max:{max_w}\tH: Estimated: {h_tmp}\t Expected: {h}')
                df_h.loc[min_w, max_w] = h_estimated
                df_rmse.loc[min_w, max_w] = np.sqrt(rmse / n_sims)
        Path.mkdir(curr_dir.joinpath('results'), exist_ok=True)
        df_rmse.to_csv(f'{curr_dir}/results/H_{h}-rmse.csv', index=True)
        df_h.to_csv(f'{curr_dir}/results/H_{h}-dist.csv', index=True)
    
    with Pool() as pool:
        pool.map(your_function, iter(h_vec))


    """
    
    def h_over_fractional(h_index, min_w, max_w):
    mock_params = [1, 0.00, 0.05, 0.10, h_index]  # (S0, mu, theta, sigma, H)
    fBM = FractionalBM()
    fBM.set_parameters(mock_params)
    sims_df = fBM.simulate(n_sims=1, t_steps=365 * 5, dt=1 / 365)
    noise = np.squeeze(sims_df.values)
    h = h_estimate_func(noise, 'price', min_window=min_w, max_window=max_w)
    return h #max(min(h, 1), 0)
    
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
    """


