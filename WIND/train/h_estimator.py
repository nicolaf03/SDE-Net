import argparse
import torch
import pathlib
from hurst import compute_Hc
import pandas as pd
from WIND.data_loader import data_loader


def get_args():
    parser = argparse.ArgumentParser(description='SDE-Net Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate of drift net')
    parser.add_argument('--lr2', default=0.01, type=float, help='learning rate of diffusion net')
    # parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
    #
    parser.add_argument('--zone', default='mock', help='zone')
    parser.add_argument('--h', default=1, help='time horizon forecasting')
    parser.add_argument('--H', default=200, help='length of history')
    #
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--decreasing_lr', default=[20], nargs='+', help='decreasing strategy')
    parser.add_argument('--decreasing_lr2', default=[], nargs='+', help='decreasing strategy')

    args = parser.parse_args()
    print(args)
    return args


def h_estimate_func(args):
    # train_loader, test_loader = data_loader.getDataSet(args.zone, args.H, args.h, args.batch_size, args.test_batch_size)
    file = pathlib.Path('../data/wind_mock_train.csv')
    df = pd.read_csv(file, index_col=0, header=0)
    """
        The kind parameter of the compute_Hc function can have the following values:
        'change': a series is just random values (i.e. np.random.randn(...))
        'random_walk': a series is a cumulative sum of changes (i.e. np.cumsum(np.random.randn(...)))
        'price': a series is a cumulative product of changes (i.e. np.cumprod(1+epsilon*np.random.randn(...))
        ==> therefore: kind 'random_walk' is for ABM, 'price' for GBM
    """
    # h, _, _ = compute_Hc(train_loader.dataset.data, kind='price', simplified=False)
    h, _, _ = compute_Hc(df.energy.values, kind='random_walk', simplified=True)
    return h


if __name__ == '__main__':
    args = get_args()
    h = h_estimate_func(args)
    print(h)




