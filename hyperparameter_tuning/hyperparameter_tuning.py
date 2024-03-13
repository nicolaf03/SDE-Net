from model.gan_model import GanModel, Generator, Discriminator
from utils import csv_utils
from utils.pytorchtools import EarlyStopping

import ray
from ray import train, tune
# from ray.air import session
# from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import AsyncHyperBandScheduler

from datetime import datetime
import numpy as np
import tqdm
import os
import time
import torch
import torch.optim.swa_utils as swa_utils
from pathlib import Path

curr_dir = Path(__file__).parent


class MyTrainableClass(tune.Trainable):
    def setup(self, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        #
        zone = params['custom']['zone']
        # log = init_log('gan_model', curr_dir / '..' / 'logs' / (zone + '_hyperparameters_tuning.log'))
        # log.info(f'zone = {zone}')
        # log.info('loading model...')
        model = GanModel(params=params, log_name='gan_model')
        
        # log.info('loading training data...')
        #model.load_training_data(curr_dir / '..' / 'data')
        #! ABSOLUTE PATH
        data_file_path = f'~/Documents/GitHub/SDE-Net/data/res_{zone.upper()}.csv'
        model.data = csv_utils.load_data(data_file_path)
        
        if model.train_params is None:
            model.train_params = dict()
        model.train_params['training_data'] = f'res_{zone.upper()}.csv'
        
        #! ABSOLUTE PATH
        models_folder = '../../../../trained_models'
        model_name = params['custom']['name']
        model.load_model(models_folder, model_name, self.device)
        
        self.model = model
        
        # Models to device
        self.model.generator = self.model.generator.to(self.device)
        self.model.discriminator = self.model.discriminator.to(self.device)
        
        # Picking a good initialisation
        with torch.no_grad():
            for param in self.model.generator._initial.parameters():
                param *= params['gan']['init_mult1']
            for param in self.model.generator._func.parameters():
                param *= params['gan']['init_mult2']
        
        # DataLoader
        #
        start_train, end_train = params['custom']['train_window']
        start_valid, end_valid = params['custom']['valid_window']
        
        df_train = self.model.data.loc[start_train:end_train]
        df_valid = self.model.data.loc[start_valid:end_valid]
        
        # remove valid_set from train_set
        index_diff = df_train.index.difference(df_valid.index)
        df_train = df_train.loc[index_diff]
        
        if self.model.ts is None or self.model.train_dataloader is None:
            self.model.ts, self.model.train_dataloader = self.model._create_dataloader(df_train)
            self.model.ts = self.model.ts.to(self.device)
        if self.model.valid_dataloader is None:
            _, self.model.valid_dataloader = self.model._create_dataloader(df_valid)
        
        # Optimizer
        #
        self.generator_optimiser = torch.optim.Adadelta(
            self.model.generator.parameters(), 
            lr=params['gan']['generator_lr'],
            weight_decay=params['gan']['weight_decay']
        )
        self.discriminator_optimiser = torch.optim.Adadelta(
            self.model.discriminator.parameters(), 
            lr=params['gan']['discriminator_lr'],
            weight_decay=params['gan']['weight_decay']
        )
        
        # Early stopping
        #
        #! ABSOLUTE PATH
        path = f'./train/checkpoints/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        if not os.path.exists(path):
            os.makedirs(path)
        self.early_stopping = EarlyStopping(
            patience=self.model.custom_params['patience'],
            path=path,
            verbose=True
        )


    
    def step(self):
        # Training logic for one iteration/epoch
        infinite_train_dataloader = (elem for it in iter(lambda: self.model.train_dataloader, None) for elem in it)
        
        train_losses = []
        valid_losses = []
        
        ###################
        # train the model #
        ###################
        self.model.generator.train(), self.model.discriminator.train()
        for step in range(self.model.custom_params['steps']):
            real_samples, = next(infinite_train_dataloader)
            real_samples = real_samples.to(self.device)
            
            self.generator_optimiser.zero_grad()
            self.discriminator_optimiser.zero_grad()

            generated_samples = self.model.generator(self.model.ts, self.model.custom_params['batch_size'])
            generated_samples = generated_samples.to(self.device)
            
            generated_score = self.model.discriminator(generated_samples)
            generated_score = generated_score.to(self.device)
            
            real_score = self.model.discriminator(real_samples)
            real_score = real_score.to(self.device)
            
            loss = generated_score - real_score
            loss = loss.to(self.device)
            loss.backward()
            train_losses.append(loss.item())

            for param in self.model.generator.parameters():
                param.grad *= -1
            
            self.generator_optimiser.step()
            self.discriminator_optimiser.step()
        
        ######################    
        # validate the model #
        ######################
        self.model.generator.eval(), self.model.discriminator.eval()
        for real_samples in self.model.valid_dataloader:
            real_samples = real_samples[0]
            real_samples = real_samples.to(self.device)
            
            generated_samples = self.model.generator(self.model.ts, self.model.custom_params['batch_size'])
            generated_samples = generated_samples.to(self.device)
            
            generated_score = self.model.discriminator(generated_samples)
            generated_score = generated_score.to(self.device)
            
            real_score = self.model.discriminator(real_samples)
            real_score = real_score.to(self.device)
            
            loss = generated_score - real_score
            valid_losses.append(loss.item())
        
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        
        # self.model.generator.load_state_dict(self.averaged_generator.module.state_dict())
        # self.model.discriminator.load_state_dict(self.averaged_discriminator.module.state_dict())
        
        # Return metrics
        return {"wasserstein_1d": valid_loss}


    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = {
            "generator_state_dict": self.model.generator.state_dict(),
            "discriminator_state_dict": self.model.discriminator.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimiser.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimiser.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint


    def load_checkpoint(self, checkpoint):
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.generator_optimiser.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        self.discriminator_optimiser.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])



if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    #device = 'cuda:0' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    else:
        print("CUDA is available")
    
    zone = 'SUD'
    
    # Set up the hyperparameter search space
    params = {
            "custom": {
                "name": f"{zone}_v2",
                "zone": zone,
                "train_window": ('2015-01','2020-12'),
                "valid_window": ('2021-01','2021-09'),
                "t_size": 7,
                "batch_size": 16,#tune.choice([32, 64, 128]),
                "steps": 200,
                "patience": 1000,
                "swa_step_start": 2500,
                "steps_per_print": 5,
            },
            "gan": {
                "initial_noise_size": 5,#tune.qrandint(5, 10),
                "noise_size": 3,#tune.qrandint(3, 10),
                "hidden_size": 16,#tune.qrandint(16, 32),
                "mlp_size": 16,#tune.qrandint(16, 32),
                "num_layers": 1,
                "generator_lr": tune.loguniform(1e-5, 1e-2),
                "discriminator_lr": tune.loguniform(1e-4, 1e-1),
                "init_mult1": 3,#tune.loguniform(1e-2, 3),
                "init_mult2": 0.5,#tune.loguniform(1e-2, 1),
                "weight_decay": 0.01
            }
    }
    
    try:
        ray.init()
    except Exception:
        pass  # already initialized
    
    # TuneConfig
    current_best_params = [
        {
            "custom/name": f"{zone}_v2",
            "custom/zone": zone,
            "custom/train_window": ('2015-01','2020-12'),
            "custom/valid_window": ('2021-01','2021-09'),
            "custom/t_size": 7,
            "custom/batch_size": 16,#tune.choice([32, 64, 128]),
            "custom/steps": 200,
            "custom/patience": 1000,
            "custom/swa_step_start": 2500,
            "custom/steps_per_print": 5,
            #
            "gan/initial_noise_size": 5,#tune.qrandint(5, 10),
            "gan/noise_size": 3,#tune.qrandint(3, 10),
            "gan/hidden_size": 16,#tune.qrandint(16, 32),
            "gan/mlp_size": 16,#tune.qrandint(16, 32),
            "gan/num_layers": 1,
            "gan/generator_lr": 2e-4,
            "gan/discriminator_lr": 1e-3,
            "gan/init_mult1": 3,
            "gan/init_mult2": 0.5,
            "gan/weight_decay": 0.01
        }
    ]
    tune_config = tune.TuneConfig(
        metric='wasserstein_1d',
        mode='min',
        search_alg=HyperOptSearch(
            metric='wasserstein_1d', 
            mode='min',
            points_to_evaluate=current_best_params
        ),
        scheduler=AsyncHyperBandScheduler(
            # metric='wasserstein_1d',
            # mode='min',
            max_t=15
        ), 
        num_samples=100,  # Number of different hyperparameter combinations
        #max_concurrent_trials=5,  # Maximum concurrent trials
    )
    
    # RunConfig
    run_config = train.RunConfig(
        name='SUD_learning_rate_tuning',
        #! ABSOLUTE PATH
        local_dir='/Users/nicolafraccarolo/Documents/GitHub/SDE-Net/hyperparameter_tuning/ray_results',
        stop={
            "training_iteration": 100,
            "time_total_s": 200
        },
        verbose=1
    )
    
    # Trainable
    #trainable = tune.with_resources(MyTrainableClass, resources={"cpu": 4})
    trainable = MyTrainableClass
    
    # Create a Tuner and run it
    tuner = tune.Tuner(
        trainable,
        param_space=params,
        tune_config=tune_config,
        run_config=run_config,
    )
    results = tuner.fit()
    
    df_results = results.get_dataframe()
    df_results.to_csv(curr_dir / 'best_params.csv')
    


    
