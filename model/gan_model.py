from pathlib import Path
import torch
import torch.optim.swa_utils as swa_utils
from utils.pytorchtools import EarlyStopping
import pandas as pd
import numpy as np
import torchcde
import torchsde
import json
import logging
from datetime import datetime, date
from matplotlib import pyplot as plt
import os
import time
import tqdm

from dtaidistance import dtw
from utils.mmd_loss import MMDLoss

import wandb
os.environ['WANDB_MODE'] = 'online'
os.environ['WANDB__SERVICE_WAIT'] = '30'

import utils.math_utils as math_utils
import utils.csv_utils as csv_utils
from utils.log_utils import init_log

from model.generator import Generator
from model.discriminator import Discriminator
from plot.plot_prediction import plot_hist, plot_samples

curr_dir = Path(__file__).parent


class GanModel:
    def __init__(self, zone=None, params=None, log_name=None):
        self.data = None
        #
        self.ts = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        #
        self.trained_generator = None
        self.trained_discriminator = None
        #
        self.train_params = None
        self.backup_data = None

        if log_name is None:
            self.log_name = 'gan_model'
            init_log(self.log_name, log_file=None, mode='w+')
        else:
            self.log_name = log_name

        if params is not None:
            self.custom_params = params['custom']
            self.gan_params = params['gan']
        else:
            if zone is None:
                raise ValueError('you should set the params or the zone!')

            self.custom_params = {
                'zone': zone,
                'steps': 10000,
                'swa_step_start': 5000,
                't_size': 64,
                'batch_size': 512,
                'steps_per_print': 10,
                'num_plot_samples': 50,
                'plot_locs': (0.1, 0.3, 0.5, 0.7, 0.9),
            }
            self.gan_params = {
                'initial_noise_size': 5,
                'noise_size': 3,
                'hidden_size': 16,
                'mlp_size': 16,
                'num_layers': 1,
                'generator_lr': 2e-4,
                'discriminator_lr': 1e-3,
                'init_mult1': 3,
                'init_mult2': 0.5,
                'weight_decay': 0.01
            }
    
    
    def load_training_data(self, historic_folder):
        self.data = GanModel._load_historic_data(historic_folder, self.custom_params['zone'])
        zone = self.custom_params['zone']
        if self.train_params is None:
            self.train_params = dict()
        self.train_params['training_data'] = f'res_{zone.upper()}.csv'
        self.ts = None
        self.train_dataloader = None
    
    
    @staticmethod
    def _load_historic_data(folder, zone):
        filename = f'res_{zone.upper()}.csv'
        data_file_path = folder / filename
        df = csv_utils.load_data(data_file_path)
        print(f'Historical data are from: {df["date"].min()} - to: {df["date"].max()}')
        return df
    
    
    def _create_dataloader(self, data):
        t_size = self.custom_params['t_size']
        batch_size = self.custom_params['batch_size']

        ts = torch.linspace(0, t_size - 1, t_size)
        
        value_array = np.array(data.iloc[:,1], dtype='float32')
        values = []
        for i in range(len(data)-t_size):
            sub_array = value_array[i:i+t_size]
            x = torch.from_numpy(np.expand_dims(sub_array,0))
            values.append(x)
        ys = torch.stack(values).transpose(1,2)
        
        dataset_size = ys.shape[0]
        
        ###################
        # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
        # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
        # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
        ###################
        y0_flat = ys[0].view(-1)
        y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat)) #? unnecessary
        ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()
        
        # todo: do it without loop
        # make all paths start from 0
        for i in range(ys.size()[0]):
            ys[i] = ys[i] - ys[:,0,:][i]
            
        ###################
        # Time must be included as a channel for the discriminator.
        ###################
        ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1), ys], dim=2)
        
        ###################
        # Package up
        ###################
        # data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
        ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
        dataset = torch.utils.data.TensorDataset(ys_coeffs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return ts, dataloader
    
    
    def _create_test_dataloader(self, start=None, end=None):
        data = self.data
        if start is not None:
            data = data[start:]
        if end is not None:
            data = data[:end]
            
        t_size = self.custom_params['t_size']
        batch_size = self.custom_params['batch_size']

        ts = torch.linspace(0, t_size - 1, t_size)
        
        value_array = np.array(data.iloc[:,1], dtype='float32')
        values = []
        for i in range(self.custom_params['num_plot_samples']):
            x = torch.from_numpy(np.expand_dims(value_array,0))
            values.append(x)
        ys = torch.stack(values).transpose(1,2)
        
        dataset_size = ys.shape[0]
        
        ###################
        # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
        # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
        # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
        ###################
        y0_flat = ys[0].view(-1)
        y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat)) #? unnecessary
        ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()
        
        # todo: do it without loop
        # make all paths start from 0
        for i in range(ys.size()[0]):
            ys[i] = ys[i] - ys[:,0,:][i]
            
        ###################
        # Time must be included as a channel for the discriminator.
        ###################
        ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1), ys], dim=2)
        
        ###################
        # Package up
        ###################
        # data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
        ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
        dataset = torch.utils.data.TensorDataset(ys_coeffs)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return ts, dataloader

    
    @staticmethod
    def _evaluate_loss(ts, batch_size, dataloader, generator, discriminator, device):
        with torch.no_grad():
            total_samples = 0
            total_loss = 0
            for real_samples, in dataloader:
                generated_samples = generator(ts, batch_size).to(device)
                generated_score = discriminator(generated_samples)
                real_score = discriminator(real_samples.to(device))
                loss = generated_score - real_score
                total_samples += batch_size
                total_loss += loss.item() * batch_size
        return total_loss / total_samples
    
    
    def _train(self, device):
        log = logging.getLogger(self.log_name)
        
        ts = self.ts.to(device)
        train_dataloader = self.train_dataloader
        infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

        # unwrap gan hyperparameters
        # todo: passare alle classi direttamente il dizionario
        init_mult1 = self.gan_params['init_mult1']
        init_mult2 = self.gan_params['init_mult2']
        generator_lr = self.gan_params['generator_lr']
        discriminator_lr = self.gan_params['discriminator_lr']
        weight_decay = self.gan_params['weight_decay']
        swa_step_start = self.custom_params['swa_step_start']
        batch_size = self.custom_params['batch_size']
        
        # Models
        generator = Generator(self.gan_params).to(device)
        discriminator = Discriminator(self.gan_params).to(device)
        
        # Weight averaging really helps with GAN training.
        averaged_generator = swa_utils.AveragedModel(generator)
        averaged_discriminator = swa_utils.AveragedModel(discriminator)
        
        # Picking a good initialisation
        with torch.no_grad():
            for param in generator._initial.parameters():
                param *= init_mult1
            for param in generator._func.parameters():
                param *= init_mult2
        
        # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
        generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=generator_lr, weight_decay=weight_decay)
        discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=discriminator_lr,
                                                    weight_decay=weight_decay)

        # Train both generator and discriminator.
        trange = tqdm.tqdm(range(self.custom_params['steps']))
        wandb.init(project='wind_gan')
        for step in trange:
            real_samples, = next(infinite_train_dataloader)
            real_samples = real_samples.to(device)

            generated_samples = generator(ts, batch_size)
            generated_samples = generated_samples.to(device)
            
            generated_score = discriminator(generated_samples)
            generated_score = generated_score.to(device)
            
            real_score = discriminator(real_samples)
            real_score = real_score.to(device)
            
            loss = generated_score - real_score
            loss = loss.to(device)
            
            wandb.log({'train loss': loss})
            
            loss.backward()

            for param in generator.parameters():
                param.grad *= -1
            generator_optimiser.step()
            discriminator_optimiser.step()
            generator_optimiser.zero_grad()
            discriminator_optimiser.zero_grad()

            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            with torch.no_grad():
                for module in discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)

            # Stochastic weight averaging typically improves performance.
            if step > swa_step_start:
                averaged_generator.update_parameters(generator)
                averaged_discriminator.update_parameters(discriminator)

            if (step % self.custom_params['steps_per_print']) == 0 or step == self.custom_params['steps'] - 1:
                total_unaveraged_loss = GanModel._evaluate_loss(
                    ts, 
                    batch_size, 
                    train_dataloader,
                    generator, 
                    discriminator, 
                    device
                )
                if step > swa_step_start:
                    total_averaged_loss = GanModel._evaluate_loss(
                        ts, 
                        batch_size,
                        train_dataloader,
                        averaged_generator.module,
                        averaged_discriminator.module, 
                        device
                    )
                    trange.write(
                        f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                        f"Loss (averaged): {total_averaged_loss:.4f}"
                    )
                else:
                    trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
        generator.load_state_dict(averaged_generator.module.state_dict())
        discriminator.load_state_dict(averaged_discriminator.module.state_dict())
        
        return generator, discriminator
    
    
    def _train2(self, device):
        log = logging.getLogger(self.log_name)
        
        ts = self.ts.to(device)
        train_dataloader = self.train_dataloader
        infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)
        valid_dataloader = self.valid_dataloader
        infinite_valid_dataloader = (elem for it in iter(lambda: valid_dataloader, None) for elem in it)
        
        # unwrap gan hyperparameters
        # todo: passare alle classi direttamente il dizionario
        init_mult1 = self.gan_params['init_mult1']
        init_mult2 = self.gan_params['init_mult2']
        generator_lr = self.gan_params['generator_lr']
        discriminator_lr = self.gan_params['discriminator_lr']
        weight_decay = self.gan_params['weight_decay']
        swa_step_start = self.custom_params['swa_step_start']
        batch_size = self.custom_params['batch_size']
        
        # Models
        generator = Generator(self.gan_params).to(device)
        discriminator = Discriminator(self.gan_params).to(device)
        
        # Weight averaging really helps with GAN training.
        averaged_generator = swa_utils.AveragedModel(generator)
        averaged_discriminator = swa_utils.AveragedModel(discriminator)
        
        # Picking a good initialisation
        with torch.no_grad():
            for param in generator._initial.parameters():
                param *= init_mult1
            for param in generator._func.parameters():
                param *= init_mult2
        
        # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
        generator_optimiser = torch.optim.Adadelta(
            generator.parameters(), 
            lr=generator_lr, 
            weight_decay=weight_decay
        )
        discriminator_optimiser = torch.optim.Adadelta(
            discriminator.parameters(), 
            lr=discriminator_lr,
            weight_decay=weight_decay
        )
        

        # Train both generator and discriminator.
        #
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
        
        # initialize the early_stopping object
        path = f'./train/checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(
            patience=self.custom_params['patience'],
            path=path,
            verbose=True
        )
        
        n_epochs = self.custom_params['n_epochs']
        #trange = tqdm.tqdm(range(self.custom_params['steps']))
        wandb.init(project='wind_gan')
        for epoch in range(n_epochs):
            
            ###################
            # train the model #
            ###################
            generator.train(), discriminator.train()
            for step in range(self.custom_params['steps']):
                real_samples, = next(infinite_train_dataloader)
                real_samples = real_samples.to(device)
                
                generator_optimiser.zero_grad()
                discriminator_optimiser.zero_grad()
                
                generated_samples = generator(ts, batch_size)
                generated_samples = generated_samples.to(device)
                
                generated_score = discriminator(generated_samples)
                generated_score = generated_score.to(device)
                
                real_score = discriminator(real_samples)
                real_score = real_score.to(device)
                
                loss = generated_score - real_score
                loss = loss.to(device)
                loss.backward()
                train_losses.append(loss.item())
                
                for param in generator.parameters():
                    param.grad *= -1
                
                generator_optimiser.step()
                discriminator_optimiser.step()
             
               
            ######################    
            # validate the model #
            ######################
            generator.eval(), discriminator.eval()
            for real_samples in valid_dataloader:
                real_samples = real_samples[0]
                real_samples = real_samples.to(device)
                
                generated_samples = generator(ts, batch_size)
                generated_samples = generated_samples.to(device)
                
                generated_score = discriminator(generated_samples)
                generated_score = generated_score.to(device)
                
                real_score = discriminator(real_samples)
                real_score = real_score.to(device)
                
                loss = generated_score - real_score
                valid_losses.append(loss.item())
            
            
            # print training/validation statistics 
            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            avg_train_losses.append(train_loss)
            valid_loss = np.average(valid_losses)
            avg_valid_losses.append(valid_loss)
            
            epoch_len = len(str(n_epochs))
            
            wandb.log(
                {
                    'train loss': train_loss,
                    'valid loss': valid_loss
                 }
            )
            print_msg = (
                f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                f'train_loss: {train_loss:.5f} ' +
                f'valid_loss: {valid_loss:.5f}'
            )
            print(print_msg)
            
            # clear lists to track next epoch
            train_losses = []
            valid_losses = []
            
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            early_stopping(valid_loss, generator, discriminator)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        #run.finish()
        
        # load the last checkpoint with the best model
        generator.load_state_dict(torch.load(f'{path}/generator.pt'))
        discriminator.load_state_dict(torch.load(f'{path}/discriminator.pt'))
        
        return generator, discriminator, avg_train_losses, avg_valid_losses
    
    
    def train(self, device, train_window, test_window, valid_window=None, plot=False):
        log = logging.getLogger(self.log_name)
        if self.data is None:
            raise ValueError('you must call \'load_training_data\' before \'train\' !')
        
        # test_month_date = datetime.strptime(test_month, '%Y-%m')
        if valid_window is None:
            raise ValueError('validation month is \'None\' !')
            
        start_train, end_train = train_window
        start_valid, end_valid = valid_window
        start_test, end_test = test_window
        
        df_train = self.data.loc[start_train:end_train]
        df_valid = self.data.loc[start_valid:end_valid]

        # remove valid_set from train_set
        index_diff = df_train.index.difference(df_valid.index)
        df_train = df_train.loc[index_diff]
        
        if self.ts is None or self.train_dataloader is None:
            self.ts, self.train_dataloader = self._create_dataloader(df_train)
        if self.valid_dataloader is None:
            _, self.valid_dataloader = self._create_dataloader(df_valid)

        log.info(f'validation month: {start_valid} - {end_valid}')
        
        print(f"Train data from: {df_train.index.min()} to: {df_train.index.max()}")
        print(f"Valid data from: {df_valid.index.min()} to: {df_valid.index.max()}")
        
        print(f"Final data used: train {len(list(self.train_dataloader))} - valid {len(list(self.valid_dataloader))}")

        if self.train_params is None:
            self.train_params = dict()
        self.train_params['start_train'] = start_train
        self.train_params['end_train'] = end_train
        self.train_params['start_valid'] = start_valid
        self.train_params['end_valid'] = end_valid
        self.train_params['start_test'] = start_test
        self.train_params['end_test'] = end_test
        
        self.train_params['trained_on'] = str(date.today())

        #self.trained_generator, self.trained_discriminator = self._train(device)
        self.trained_generator, self.trained_discriminator, train_loss, valid_loss = self._train2(device)
        
        # if plot:
        #     lgb.plot_metric(result, metric=self.lgb_params['metric'], title=f"train metric {self.custom_params['zone']}")
        #     pyplot.vlines(x = self.trained_model.best_iteration, ymin=0, ymax=0.3, colors="red")


    # def cross_validation(self, start_train, start_test, end_test, valid_strategy=None, verbose=False,
    #                      use_tension_coeff=True, weights_strategy=None, ignore_d7=False):
    #     if self.data is None:
    #         raise ValueError('you must call \'load_training_data\' before \'train\' !')

    #     log = logging.getLogger(self.log_name)
    #     if self.ts is None or self.train_dataloader is None:
    #         self.ts, self.train_dataloader = self._create_dataloader()

    #     groups_year = set(self.features.index.year)
    #     groups_month = set(self.features.index.month)
    #     months_list = list(itertools.product(groups_year, groups_month))
    #     ignore_d7_index = pd.Index([])
    #     models = []
    #     predictions = pd.Series()
    #     gt = pd.Series()
    #     festive_days = pd.Series()
    #     bank_holidays = pd.Series()
    #     start_train_month = (int(start_train.split('-')[0]), int(start_train.split('-')[1]))
    #     start_test_month = (int(start_test.split('-')[0]), int(start_test.split('-')[1]))
    #     end_test_month = (int(end_test.split('-')[0]), int(end_test.split('-')[1]))
    #     start = False
    #     for idx, curr_month in enumerate(months_list):
    #         if curr_month == start_test_month:
    #             start = True
    #         if end_test_month == months_list[idx - 1]:
    #             break
    #         if start:
    #             end_train_month = months_list[idx - 2]
    #             test_month = curr_month
    #             # validate on the same test_month, one year before
    #             if valid_strategy is None:
    #                 # default strategy
    #                 if test_month[1] in self.custom_params['validate_prev_year']:
    #                     month_valid_strategy = 'prev_year'
    #                 else:
    #                     month_valid_strategy = 'prev_prev_month'
    #             else:
    #                 month_valid_strategy = valid_strategy

    #             if month_valid_strategy == 'prev_year' and f'{test_month[0] - 1}-{test_month[1]}' in self.features.index:
    #                 valid_month = (test_month[0] - 1, test_month[1])
    #             else:
    #                 valid_month = end_train_month

    #             if verbose:
    #                 log.info(f'Training on Period: {start_train_month}-{end_train_month}, '
    #                          f'Validate on Period: {valid_month}, '
    #                          f'Test on Period: {test_month}')

    #             df_train = self.features[start_train:f'{end_train_month[0]}-{end_train_month[1]}']
    #             df_valid = self.features[f'{valid_month[0]}-{valid_month[1]}']
    #             df_test = self.features[f'{test_month[0]}-{test_month[1]}']

    #             # save festive days target and remove them from df_train
    #             df_train = self._save_festive_days_target(df_train)
    #             # df_train = self._save_bank_holidays_target(df_train)

    #             # remove valid_set from train_set
    #             index_diff = df_train.index.difference(df_valid.index)
    #             df_train = df_train.loc[index_diff]

    #             y_train = df_train.target
    #             x_train = df_train.loc[:, feat_names]
    #             y_valid = df_valid.target
    #             x_valid = df_valid.loc[:, feat_names]
    #             y_test = df_test.target
    #             x_test = df_test.loc[:, feat_names]

    #             if weights_strategy is None:
    #                 weights_strategy = self.custom_params['weights_strategy']
    #             train_weights = self._apply_weights_strategy(weights_strategy, df_train.index, test_month[1])

    #             if ignore_d7:
    #                 if verbose:
    #                     log.info('train without d7')
    #                 model = self._train_without_d7(features_without_d7, cat_feat_without_d7, df_train,
    #                                                           df_valid, train_weights)
    #                 test_prediction = pd.Series(model.predict(x_test[features_without_d7]), index=df_test.index)
    #             else:
    #                 model, result = self._train(
    #                     x_train, y_train, x_valid, y_valid, feat_names, cat_feat,
    #                     early_stopping_rounds=self.custom_params['early_stopping_rounds'],
    #                     num_iterations=self.custom_params['max_iterations'],
    #                     train_weights=train_weights)
    #                 test_prediction = pd.Series(model.predict(x_test), index=df_test.index)

    #                 ignore_d7_ix = self._ignore_d7_index(df_test)
    #                 ignore_d7_index = ignore_d7_index.union(ignore_d7_ix)

    #                 if len(ignore_d7_ix) > 0:
    #                     if verbose:
    #                         log.info('train without d7')
    #                     model_without_d7 = self._train_without_d7(features_without_d7, cat_feat_without_d7, df_train, df_valid, train_weights)
    #                     test_prediction.loc[ignore_d7_ix] = pd.Series(model_without_d7.predict(
    #                         df_test.loc[ignore_d7_ix, features_without_d7]), index=ignore_d7_ix)

    #             test_prediction, festive_day_no_weekend = self._set_festive_days_target(df_test, test_prediction)
    #             # test_prediction, bank_holidays_ = self._set_bank_holidays_target(df_test, test_prediction)
    #             test_prediction, bank_holidays_ = self._apply_bank_holidays_strategy(df_test, test_prediction)
    #             bank_holidays = bank_holidays.append(bank_holidays_)
    #             # un-normalize
    #             y_test = (y_test - 1.0) / 1000 * df_test['power']
    #             test_prediction = (test_prediction - 1.0) / 1000 * df_test['power']
    #             # y_test = (y_test - 1.0) * df_test['active']
    #             # test_prediction = (test_prediction - 1.0) * df_test['active']

    #             # apply tension coeff
    #             if use_tension_coeff:
    #                 tension_coeff = self.compute_tension_coeff()
    #                 if tension_coeff is not None:
    #                     test_prediction *= tension_coeff
    #                     y_test *= tension_coeff
    #                 else:
    #                     log.error('missing tension level in test data!')

    #             predictions = predictions.append(test_prediction)
    #             gt = gt.append(y_test)
    #             festive_days = festive_days.append(festive_day_no_weekend)

    #             models.append(model)

    #     return models, predictions, gt, festive_days, bank_holidays, ignore_d7_index


    def predict(self, test_window, device, plot):

        if self.trained_generator is None or self.trained_discriminator is None:
            raise ValueError('you must train or load the model before!')
        if self.data is None:
            raise ValueError('you must call \'load_training_data\' before \'train\' !')

        log = logging.getLogger(self.log_name)
        start_test, end_test = test_window
        df_test = self.data[start_test:end_test]
        if self.ts is None:
            self.ts, self.test_dataloader = self._create_dataloader(df_test)
            
        ts = self.ts.to(device)
        test_dataloader = self.test_dataloader
        #infinite_test_dataloader = (elem for it in iter(lambda: test_dataloader, None) for elem in it)
        generator = self.trained_generator
        discriminator = self.trained_discriminator
        
        real_samples = []
        generated_samples = []
        err_dtw = []
        #err_mmd = []
        mmd = MMDLoss()
        # Get samples
        for real_sample in test_dataloader:
            real_sample = real_sample[0]
            real_sample = real_sample.to(device)
            #assert self.custom_params['num_plot_samples'] <= real_samples.size(0)
            real_sample = torchcde.LinearInterpolation(real_sample).evaluate(ts)
            with torch.no_grad():
                real_score = discriminator(real_sample)
            real_sample = real_sample[..., 1]
            real_samples.append(real_sample)

            with torch.no_grad():
                generated_samples_it = []
                for i in range(3):
                    generated_sample = generator(ts, real_sample.size(0)).to(device)
                    generated_samples_it.append(generated_sample)
                generated_sample = torch.mean(torch.stack(generated_samples_it, dim=0), dim=0)
                #generated_sample = generator(ts, real_sample.size(0)).to(device)
                generated_score = discriminator(generated_sample)
            generated_sample = torchcde.LinearInterpolation(generated_sample).evaluate(ts)
            generated_sample = generated_sample[..., 1]
            generated_samples.append(generated_sample)
            
            for real_s, generated_s in zip(real_samples, generated_samples):
                for i in range(real_s.size()[0]):
                    err_dtw.append(dtw.distance(real_s[i,:].numpy(), generated_s[i,:].numpy()))
                    #err_mmd.append(mmd(real_s[i,:], generated_s[i,:]).item())
            #loss = generated_score - real_score
            #err.append(loss.item())
            
        real_samples = torch.cat(real_samples, dim=0)
        generated_samples = torch.cat(generated_samples, dim=0)
        
        if plot:
            plot_hist(real_samples, generated_samples, self.custom_params['plot_locs'], self.custom_params['zone'])
            plot_samples(ts, real_samples, generated_samples, self.custom_params['num_plot_samples'], self.custom_params['zone'])
            #plt.show()
        
        mean_err_dtw = np.mean(err_dtw)
        err_mmd = mmd(real_samples, generated_samples).item()
        mean_err = {'DTW': mean_err_dtw, 'MMD': err_mmd}
        return generated_samples, real_samples, mean_err
    
    
    def predict2(self, start_test, end_test, device, plot):

        if self.trained_generator is None or self.trained_discriminator is None:
            raise ValueError('you must train or load the model before!')
        if self.data is None:
            raise ValueError('you must call \'load_training_data\' before \'train\' !')

        log = logging.getLogger(self.log_name)
        #self.ts, self.train_dataloader = self._create_dataloader(end=start_test)
        self.ts, test_dataloader = self._create_test_dataloader(start=start_test, end=end_test)
        ts = self.ts.to(device)
        
        generator = self.trained_generator

        # Get samples
        real_samples, = next(iter(test_dataloader))
        real_samples = real_samples.to(device)
        assert self.custom_params['num_plot_samples'] <= real_samples.size(0)
        real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
        real_samples = real_samples[..., 1]

        with torch.no_grad():
            generated_samples = generator(ts, real_samples.size(0)).to(device)
        generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
        generated_samples = generated_samples[..., 1]
        
        if plot:
            #plot_hist(real_samples, generated_samples, self.custom_params['plot_locs'], self.custom_params['zone'])
            plot_samples(ts, real_samples, generated_samples, self.custom_params['num_plot_samples'], self.custom_params['zone'])
            plt.show()
        
        y_test = None
        mean_err = None
        return generated_samples, real_samples
    

    @staticmethod
    def get_params(name, folder='parameters'):
        params_filename = 'params_' + name + '.json'
        with open(Path(folder) / params_filename) as json_file:
            params = json.load(json_file)
        return params
    

    @staticmethod
    def load_params(folder, name, log_name=None):
        params_filename = 'params_' + name + '.json'
        with open(Path(folder) / params_filename) as json_file:
            params = json.load(json_file)

        return GanModel(params=params, log_name=log_name)
    
    
    # def save_params(self, folder, name):
    #     params_filename = 'params_' + name + '.json'
    #     with open(Path(folder) / params_filename, 'w') as outfile:
    #         json.dump({'custom': self.custom_params, 'lgb': self.lgb_params}, outfile)

    #     return Path(folder) / params_filename
    
    
    @staticmethod
    def filenames(name):
        generator_filename = f'generator_{name}'
        discriminator_filename = f'discriminator_{name}'
        train_params_filename = f'train_{name}.json'
        return [generator_filename, discriminator_filename, train_params_filename]
    
    
    def save_model(self, folder, name):
        if self.trained_generator is None or self.trained_discriminator is None:
            raise ValueError('you must train or load the model before!')

        [generator_filename, discriminator_filename, train_params_filename] = self.filenames(name)

        torch.save(self.trained_generator.state_dict(), str((Path(folder) / generator_filename).resolve()))
        torch.save(self.trained_discriminator.state_dict(), str((Path(folder) / discriminator_filename).resolve()))

        # save train params
        self.train_params['model_name'] = name
        self.train_params['custom'] = self.custom_params
        self.train_params['gan'] = self.gan_params
        with open(Path(folder) / train_params_filename, 'w') as outfile:
            json.dump(self.train_params, outfile)

        return [generator_filename, discriminator_filename, train_params_filename]
    
    
    def load_model(self, folder, name, device):
        generator_filename, discriminator_filename, train_params_filename = self.filenames(name)

        if self.custom_params is None or self.gan_params is None:
            raise ValueError('you must call \'load_params\'!')
        self.trained_generator = Generator(self.gan_params)
        self.trained_discriminator = Discriminator(self.gan_params)
        
        self.trained_generator.load_state_dict(
            torch.load(str((Path(folder) / generator_filename).resolve()),
                       map_location=torch.device(device)
            )
        )
        self.trained_discriminator.load_state_dict(
            torch.load(str((Path(folder) / discriminator_filename).resolve()),
                       map_location=torch.device(device)
            )
        )

        # load train params
        with open(Path(folder) / train_params_filename) as json_file:
            self.train_params = json.load(json_file)
    


if __name__ == '__main__':
    print('hello')
