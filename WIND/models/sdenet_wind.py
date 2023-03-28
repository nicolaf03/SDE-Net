import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import math
from pathlib import Path
import json
import pandas as pd
import logging
from datetime import date

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
        
        
    
        
        
    def train(self, start_train, end_train, valid_month, plot=False):
        log = logging.getLogger(self.log_name)
        
        if self.data is None:
            raise ValueError('you must call \'load_training_data\' before \'train\' !')

        # todo: valutare di aggiungere in seguito (es: temperature)
        if self.features is None:
            self.features = self._compute_features()

        log.info(f'validation month: {valid_month}')

        if self.train_params is None:
            self.train_params = dict()
        self.train_params['start_train'] = start_train
        self.train_params['end_train'] = end_train
        self.train_params['valid_month'] = valid_month
        self.train_params['trained_on'] = str(date.today())

        df_train = self.features[start_train:end_train]
        df_valid = self.features[valid_month]

        # remove valid_set from train_set
        index_diff = df_train.index.difference(df_valid.index)
        df_train = df_train.loc[index_diff]

        print(f"Final data used: train {len(df_train)} - valid {len(df_valid)}")
        print(f"Train data from: {df_train.index.min()} to: {df_train.index.max()}")
        print(f"Valid data from: {df_valid.index.min()} to: {df_valid.index.max()}")
        
        
        # train_loader, test_loader = data_loader.getDataSet(args.zone, args.H, args.h, args.batch_size, args.test_batch_size)
        
        '''
        def _train(self, x_train, x_valid, feat_names, cat_feat, early_stopping_rounds, num_iterations):
            log = logging.getLogger(self.log_name)

            train = lgb.Dataset(x_train, label=y_train, free_raw_data=False, weight=weights,
                                feature_name=feat_names, categorical_feature=cat_feat)

            if x_valid is not None and y_valid is not None:
                valid = lgb.Dataset(x_valid, label=y_valid, reference=train, free_raw_data=False,
                                    feature_name=feat_names, categorical_feature=cat_feat)
                dataset_names = ['train', 'valid']
                datasets = [train, valid]
            else:
                dataset_names = ['train']
                datasets = [train]

            result = dict()
            model = lgb.train(self.lgb_params, train,
                            num_boost_round=num_iterations,
                            early_stopping_rounds=early_stopping_rounds,
                            categorical_feature=cat_feat,
                            valid_sets=datasets,
                            valid_names=dataset_names,
                            evals_result=result,
                            verbose_eval=False)

            return model, result
        '''
        
        
        #------------------------------

        self.trained_model, result = self._train(
            df_train.loc[:, features], df_train.target, df_valid.loc[:, features], df_valid.target,
            features, cat_feat,
            early_stopping_rounds=self.custom_params['early_stopping_rounds'],
            num_iterations=self.custom_params['max_iterations'],
            train_weights=train_weights
        )
        if plot:
            lgb.plot_metric(result, metric=self.lgb_params['metric'], title=f"train metric {self.custom_params['zone']}")
            pyplot.vlines(x = self.trained_model.best_iteration, ymin=0, ymax=0.3, colors="red")
            pyplot.show()

        # removes_also_august
        #if self.custom_params["zone"] == 'NORD':
        #    df_train = df_train.loc[~((df_train.index.month == 8) & (df_train.index.year == 2020)),]
        #    print(f"Removed 2020-08 for nod7 training {len(df_train)}")

        self.trained_model_nod7, result = self._train(
            df_train.loc[:, features_without_d7], df_train.target, df_valid.loc[:, features_without_d7], df_valid.target,
            features_without_d7, cat_feat_without_d7,
            early_stopping_rounds=self.custom_params['early_stopping_rounds'],
            num_iterations=self.custom_params['max_iterations'],
            train_weights=train_weights)
        if plot:
            lgb.plot_metric(result, metric=self.lgb_params['metric'], title=f"train without d7 metric {self.custom_params['zone']}")
            pyplot.vlines(x = self.trained_model_nod7.best_iteration, ymin=0, ymax=0.3, colors="red")

        # valid_err = result['valid'][self.lgb_params['metric']][-1]
        
        
    def _compute_features(self):
        #todo: sistema
        data = self.data
        features = data.copy()
        start_date = features.index.min()
        end_date = features.index.max()

        # consumption normalization by zone max power
        # todo

        features = features.loc[start_date:end_date]

        # todo
        # features = features.join(self.temp)
        # # fill NaNs in temps columns
        # temp_cols = list(self.temp.columns)
        # features.loc[:, temp_cols] = features[temp_cols].fillna(method='ffill').fillna(method='bfill')

        return features
    
    
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
