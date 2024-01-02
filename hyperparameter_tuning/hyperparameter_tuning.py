# import numpy as np
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from ray import train, tune
# from ray.tune.schedulers import ASHAScheduler
# from utils.log_utils import init_log, dispose_log
from model.gan_model import GanModel, Generator, Discriminator
from utils import csv_utils

from ray import tune
from ray.tune.tuner import Tuner
from ray.train import RunConfig

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
        data_file_path = f'~/Documents/GitHub/SDE-Net/data/res_{zone.upper()}.csv'
        model.data = csv_utils.load_data(data_file_path)
        if model.train_params is None:
            model.train_params = dict()
        model.train_params['training_data'] = f'res_{zone.upper()}.csv'
        model.ts = None
        model.train_dataloader = None
        
        self.model = model
        
        # Models
        self.generator = Generator(params['gan']).to(self.device)
        self.discriminator = Discriminator(params['gan']).to(self.device)
        # Weight averaging really helps with GAN training.
        self.averaged_generator = swa_utils.AveragedModel(self.generator)
        self.averaged_discriminator = swa_utils.AveragedModel(self.discriminator)
        
        # Picking a good initialisation
        with torch.no_grad():
            for param in self.generator._initial.parameters():
                param *= params['gan']['init_mult1']
            for param in self.generator._func.parameters():
                param *= params['gan']['init_mult2']
        
        # DataLoader
        #
        if self.model.ts is None or self.model.train_dataloader is None:
            self.model.ts, self.model.train_dataloader = self.model._create_dataloader()
        
        # Optimizer
        #
        self.generator_optimiser = torch.optim.Adadelta(
            self.generator.parameters(), 
            lr=params['gan']['generator_lr'],
            weight_decay=params['gan']['weight_decay']
        )
        self.discriminator_optimiser = torch.optim.Adadelta(
            self.discriminator.parameters(), 
            lr=params['gan']['discriminator_lr'],
            weight_decay=params['gan']['weight_decay']
        )
        
        
    
    def step(self):
        # Training logic for one iteration/epoch
        infinite_train_dataloader = (elem for it in iter(lambda: self.model.train_dataloader, None) for elem in it)
        # trange = tqdm.tqdm(range(self.model.custom_params['steps']))
        # for step in trange:
        for step in range(self.model.custom_params['steps']):
            real_samples, = next(infinite_train_dataloader)
            real_samples = real_samples.to(self.device)

            generated_samples = self.generator(self.model.ts, self.model.custom_params['batch_size'])
            generated_samples = generated_samples.to(self.device)
            
            generated_score = self.discriminator(generated_samples)
            generated_score = generated_score.to(self.device)
            
            real_score = self.discriminator(real_samples)
            real_score = real_score.to(self.device)
            
            loss = generated_score - real_score
            loss = loss.to(self.device)
            
            loss.backward()

            for param in self.generator.parameters():
                param.grad *= -1
            self.generator_optimiser.step()
            self.discriminator_optimiser.step()
            self.generator_optimiser.zero_grad()
            self.discriminator_optimiser.zero_grad()

            ###################
            # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
            # LipSwish activation functions).
            ###################
            with torch.no_grad():
                for module in self.discriminator.modules():
                    if isinstance(module, torch.nn.Linear):
                        lim = 1 / module.out_features
                        module.weight.clamp_(-lim, lim)

            # Stochastic weight averaging typically improves performance.
            if step > self.model.custom_params['swa_step_start']:
                self.averaged_generator.update_parameters(self.generator)
                self.averaged_discriminator.update_parameters(self.discriminator)

            if (step % self.model.custom_params['steps_per_print']) == 0 or step == self.model.custom_params['steps'] - 1:
                total_unaveraged_loss = GanModel._evaluate_loss(
                    self.model.ts, 
                    self.model.custom_params['batch_size'], 
                    self.model.train_dataloader,
                    self.generator, 
                    self.discriminator, 
                    self.device
                )
                if step > self.model.custom_params['swa_step_start']:
                    total_averaged_loss = GanModel._evaluate_loss(
                        self.model.ts, 
                        self.model.custom_params['batch_size'],
                        self.model.train_dataloader,
                        self.averaged_generator.module,
                        self.averaged_discriminator.module, 
                        self.device
                    )
                #     trange.write(
                #         f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                #         f"Loss (averaged): {total_averaged_loss:.4f}"
                #     )
                # else:
                #     trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
        self.generator.load_state_dict(self.averaged_generator.module.state_dict())
        self.discriminator.load_state_dict(self.averaged_discriminator.module.state_dict())
        
        # Return metrics
        return {"wasserstein_1d": loss}


    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


    def load_checkpoint(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))



if __name__ == '__main__':
    zone = 'SUD'
    
    # Set up the hyperparameter search space
    params={
            "custom": {
                "name": f"{zone}_model_v1",
                "zone": zone,
                "t_size": 64,
                "batch_size": tune.choice([512, 1024]),
                "steps": 10,
                "swa_step_start": 5,
                "steps_per_print": 5,
            },
            "gan": {
                "initial_noise_size": 5,
                "noise_size": 3,
                "hidden_size": 16,
                "mlp_size": 16,
                "num_layers": 1,
                "generator_lr": tune.choice([1e-3, 1e-2]),#tune.loguniform(1e-5, 1e-2),
                "discriminator_lr": 1e-2,#tune.loguniform(1e-4, 1e-1),
                "init_mult1": 1,
                "init_mult2": 1,
                "weight_decay": 0.01
            }
    }
    
    # Configure Tune
    tune_config = tune.TuneConfig(
        search_alg=tune.search.HyperOptSearch(),  # Bayesian optimization
        scheduler=tune.schedulers.ASHAScheduler(),  # Early stopping
        num_samples=1,  # Number of different hyperparameter combinations
        #max_concurrent_trials=5,  # Maximum concurrent trials
    )
    
    # Create a Tuner and run it
    tuner = Tuner(
        MyTrainableClass,
        param_space=params,
        tune_config=tune_config
    )
    results = tuner.fit()
    
    df_results = results.get_dataframe()
    df_results.to_csv(curr_dir / 'best_params.csv')
    


    
