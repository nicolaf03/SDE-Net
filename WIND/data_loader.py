from custom_dataset import CustomTimeSeriesDataset
from torchvision import transforms
from pathlib import Path
import pandas as pd
import pdb
from torch.utils.data import DataLoader

curr_dir = Path(__file__).parent

def getDataSet(zone, H, h, batch_size, test_batch_size, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print(f'Building {zone} data loader with {num_workers} workers')
    
    # transform_train = transforms.Compose([
    #         transforms.ToTensor(),
    #     ])
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    #     ])
    
    train_loader = DataLoader(
        CustomTimeSeriesDataset(zone=zone, H=H, h=h, train=True),
        batch_size=batch_size,
        shuffle=True,
        # transform=transform_train,
        num_workers=num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        CustomTimeSeriesDataset(zone=zone, H=H, h=h, train=False),
        batch_size=test_batch_size,
        shuffle=True,
        # transform=transform_test,
        num_workers=num_workers,
        drop_last=True
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = getDataSet(zone='SUD', H=28, h=1, batch_size=128, test_batch_size=1)
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        print(inputs.shape)
        print(targets.shape)
