
import pandas as pd
import numpy as np
import logging
import multiprocess as mp # Refactoring due to https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror

from pathlib import Path
from h_estimator import compute_Hc
from fractional_BM.fBM import FractionalBM

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
    h, c, data = compute_Hc(noise, kind=kind, min_window=min_window, max_window=max_window, simplified=False)
    return h, c, data


def generate_fbm_trajectories(sigma, h_index, n_sims, T:int ):
    mock_params = [1, 0.00, 0.00, sigma, h_index]  # (S0, mu, theta, sigma, H)
    fBM = FractionalBM()
    fBM.set_parameters(mock_params)
    sims_df = fBM.simulate(n_sims=n_sims, t_steps=T, dt=1 / 365)
    noise = np.squeeze(sims_df.values)
    return noise

if __name__ == '__main__':
    # INPUT
    h_vec = np.linspace(.1, .9, 9)
    sig_vec = np.linspace(0.742997902, 0.888742816, 5)
    item = []
    for i in range(0, len(h_vec)):
        for j in range(0, len(sig_vec)):
            item.append((sig_vec[j], h_vec[i]))

    # print(input)
    def compute_h(item):
        h = item[1]
        sig = item[0]
        n_sims = 1000
        freq = 365
        t_n = freq * 5
        step = 60
        min_windows = np.arange(30, t_n - 365, step)
        max_windows = np.arange(100, t_n - 1, step)

        print(f'Try with h = {h} and {sig}')

        df_h = pd.DataFrame(data=[], columns=max_windows, index=min_windows)
        df_rmse = pd.DataFrame(data=[], columns=max_windows, index=min_windows)
        df_range_win = pd.DataFrame(data=[], columns=max_windows, index=min_windows)
        noise = generate_fbm_trajectories(sig * np.sqrt(365), h, n_sims, t_n)
        logger.info(f'{n_sims} for fbm with H = {h} and {sig} sigma generated.')
        for min_w in min_windows:
            for max_w in max_windows:
                if min_w < max_w:
                    h_estimated = []
                    se = 0
                    for n in range(0, n_sims):
                        h_tmp, c_tmp, data_tmp = h_estimate_func(noise[n, :], 'price', min_window=min_w, max_window=max_w)
                        h_estimated.append(h_tmp)
                        se += (h_tmp - h) ** 2
                        logger.info(f'w_min: {min_w}\tw_max:{max_w}\tH: Estimated: {h_tmp}\t Expected: {h}')
                    df_h.loc[min_w, max_w] = h_estimated
                    df_range_win.loc[min_w, max_w] = data_tmp[0]
                    df_rmse.loc[min_w, max_w] = np.sqrt(se / n_sims)
                else:
                    pass
        Path.mkdir(curr_dir.joinpath('results'), exist_ok=True)
        df_rmse.to_csv(f'{curr_dir}/results/H_{h}_sig_{sig}-rmse.csv', index=True)
        df_range_win.to_csv(f'{curr_dir}/results/H_{h}_sig_{sig}-range.csv', index=True)
        df_h.to_csv(f'{curr_dir}/results/H_{h}_sig_{sig}-dist.csv', index=True)
    # compute_h(0.5)
    # MULTIPROCESSING
    with mp.Pool() as pool:
        pool.map(compute_h, iter(item))






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


