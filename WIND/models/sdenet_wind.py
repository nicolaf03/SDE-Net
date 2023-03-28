import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
from pathlib import Path
import json
import pandas as pd

from utils.log_utils import init_log
import utils.csv_utils as csv_utils

__all__ = ['SDENet_wind']


# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv1d):
#             init.kaiming_normal_(m.weight, mode='fan_out')
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm1d):
#             init.constant_(m.weight, 1)
#             init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal_(m.weight, std=1e-3)
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)


class Drift(nn.Module):
    def __init__(self):
        super(Drift, self).__init__()
        self.fc = nn.Linear(50, 50)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, t, x):
        out = self.fc(x)
        out = self.relu(out)
        return out   


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, 1)
        
    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
    

class SDENet_wind(nn.Module):
    def __init__(self, layer_depth, H, zone=None, params=None, log_name=None):
        super(SDENet_wind, self).__init__()
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Linear(H, 50)
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(50, 2)
        )
        self.deltat = 4./self.layer_depth
        # self.apply(init_params)
        self.sigma = 5
        
        if log_name is None:
            self.log_name = 'sde-net_model'
            init_log(self.log_name, log_file=None, mode='w+')
        else:
            self.log_name = log_name
        
        if params is not None:
            self.custom_params = params['custom']
            self.lgb_params = params['sde-net']
        else:
            if zone is None:
                raise ValueError('you should set the params or the zone!')

            self.custom_params = {
                'zone': zone,
                'max_iterations': 40,
                # 'early_stopping_rounds': 100,
                # 'validate_prev_year': [],
                # 'weights_strategy': {},
            }
            self.sdeNet_params = {
                # 'task': 'train',
                # 'objective': 'regression',
                # 'metric': 'mape',
                # 'is_unbalance': 'true',
                # 'boosting': 'gbdt',
                # 'num_leaves': 31,
                # 'feature_fraction': 1.0,
                # 'bagging_fraction': 1.0,
                # 'bagging_freq': 0,
                # 'learning_rate': 0.05,
                # 'verbose': -1,
            }
        
    
    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out)
            for i in range(self.layer_depth):
                t = 4 * float(i) / self.layer_depth
                #*
                #* STOCHASTIC DIFFERENTIAL EQUATION
                #* (Euler-Maruyama)
                out = out \
                    + self.drift(t,out) * self.deltat \
                        + diffusion_term * math.sqrt(self.deltat) * torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
            mu = final_out[:,:,0]
            sigma = F.softplus(final_out[:,:,1]) + 1e-3
            return mu, sigma
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
            sigma = final_out[:,:,0]
            return sigma
        
    
    def load_training_data(self, historic_folder):
        #todo: importare il validation set
        self.data = SDENet_wind._load_historic_data(historic_folder, self.custom_params['zone'])
        zone = self.custom_params['zone']
        if self.train_params is None:
            self.train_params = dict()
        self.train_params['training_data'] = f'wind_{zone.upper()}_train.csv'
        self.features = None
        
        
    @staticmethod
    def _load_historic_data(folder, zone):
        filename = f'wind_{zone.upper()}_train.csv'
        data_file_path = folder / filename
        df = csv_utils.load_aggregated_data(data_file_path)

        print(f'Historical data are from: {df["date"].min()} - to: {df["date"].max()}')

        all_datetimes = pd.DataFrame(pd.date_range(df['date'].min(), df['date'].max(), freq='D'), columns=['date'])
        df = all_datetimes.merge(df, on=['date'], how='outer')
        df.index = df['date']

        df.dropna(inplace=True)

        return df
    
    
    @staticmethod
    def load_params(folder, name, log_name=None):
        params_filename = 'params_' + name + '.json'
        with open(Path(folder) / params_filename) as json_file:
            params = json.load(json_file)

        return SDENet_wind(params=params, log_name=log_name)


def test():
    model = SDENet_wind(layer_depth=10, H=64)
    return model
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)
