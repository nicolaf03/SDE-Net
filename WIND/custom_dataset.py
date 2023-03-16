from torch.utils.data import Dataset
import torchvision
import torch

from typing import Any, Callable, Dict, List, Optional, Tuple
from pathlib import Path
# import os

import pandas as pd
import numpy as np


curr_dir = Path(__file__).parent

def _load_data(self, zone, H, h):
    PATH = curr_dir / '..' / 'data' / f'wind_{zone}_{"train" if self.train else "test"}.csv'
    data = pd.read_csv(PATH)
    value_array = np.array(data.iloc[:,1], dtype='float32')
    # res = []
    values = []
    targets = []
    
    for i in range(len(data)-(H+h)):
        sub_array = value_array[i:i+(H+h)]
        x = torch.from_numpy(np.expand_dims(sub_array[:-h],0)); values.append(x)
        y = torch.from_numpy(sub_array[-h:]); targets.append(y)
        # x = torch.tensor(sub_array[:-h]); values.append(x)
        # y = torch.tensor(sub_array[-h:]); targets.append(y)
        #y = torch.from_numpy(np.expand_dims(sub_array[-h:],0)); targets.append(y)
        #res.append((torch.from_numpy(sub_array[:-h]), torch.from_numpy(sub_array[-h:])))
    
    return values, targets


class CustomTimeSeriesDataset(Dataset):
    
    def __init__(
        self, 
        zone: str, 
        H: int,
        h: int,
        train: bool = True,
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        self.train = train
        self.data, self.targets = _load_data(self, zone=zone, H=H, h=h)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)
        return data, target
    
    
if __name__ == '__main__':
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dataset = CustomTimeSeriesDataset(zone='SUD', H=28, h=1)
    print(dataset[0])
    