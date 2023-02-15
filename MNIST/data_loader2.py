import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#import os
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet
from pathlib import Path
import pandas as pd

curr_dir = Path(__file__).parent

def getMNIST(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 0)
    kwargs.pop('input_size', None)
    print("Building MNIST data loader with {} workers".format(num_workers))

    transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    ds = []
    train_loader = DataLoader(
        datasets.MNIST(root='../data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=num_workers, drop_last=True
    )
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.MNIST(root='../data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=num_workers, drop_last=True
    )
    ds.append(test_loader)

    return ds



def getSVHN(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []

    train_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='train', download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform,
        ),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)

    test_loader = DataLoader(
        datasets.SVHN(
            root='../data/svhn', split='test', download=True,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]),
            target_transform=target_transform
        ),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)
    return ds


def getSEMEION(batch_size, test_batch_size, img_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SEMEION data loader with {} workers".format(num_workers))
    ds = []
    train_loader = DataLoader(
        datasets.SEMEION(
            root='../data/semeion', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)
    ds.append(train_loader)
    test_loader = DataLoader(
        datasets.SEMEION(
            root='../data/semeion', download=True,
            transform=transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size, shuffle=False, drop_last=True, **kwargs)
    ds.append(test_loader)

    return ds



def getDataSet(zone, batch_size, test_batch_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 0)
    kwargs.pop('input_size', None)
    print(f'Building {zone} data loader with {num_workers} workers')
    
    data_path = curr_dir / '..' / 'data' / f'wind_{zone}.csv'
    data = pd.read_csv(data_path)
    
    min_encoder_length = 0
    max_encoder_length = 400     # use 252 days of history
    min_prediction_length = 1
    max_prediction_length = 7
    training_cutoff = data["time_idx"].max() - max_prediction_length
    
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx='time_idx',
        target='energy',
        group_ids=['energy'],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_prediction_length,
        max_prediction_length=max_prediction_length,
        # target_normalizer=,
        allow_missing_timesteps=True,
        time_varying_unknown_reals=['energy']
    )
    
    testing = TimeSeriesDataSet.from_dataset(
        training, data, predict=True, stop_randomization=True
    )
    
    # create dataloaders for model
    train_loader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    test_loader = testing.to_dataloader(
        train=False, batch_size=test_batch_size, num_workers=0
    )
    
    return train_loader, test_loader



if __name__ == '__main__':
    train_loader, test_loader = getDataSet('cifar10', 256, 1000, 28)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(inputs.shape)
        print(targets.shape)
