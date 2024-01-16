from model.gan_model import GanModel, Generator, Discriminator
from utils import csv_utils

from ray import train, tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

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
            self.model.ts = self.model.ts.to(self.device)
        
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
        return {"wasserstein_1d": loss.item()}


    def save_checkpoint(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        checkpoint = {
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
            "generator_optimizer_state_dict": self.generator_optimiser.state_dict(),
            "discriminator_optimizer_state_dict": self.discriminator_optimiser.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint


    def load_checkpoint(self, checkpoint):
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
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
    params={
            "custom": {
                "name": f"{zone}_model_v1",
                "zone": zone,
                "t_size": 64,
                "batch_size": tune.choice([32, 64, 128]),
                "steps": 5000,
                "swa_step_start": 2500,
                "steps_per_print": 5,
            },
            "gan": {
                "initial_noise_size": tune.qrandint(5, 10),
                "noise_size": tune.qrandint(3, 10),
                "hidden_size": tune.qrandint(16, 32),
                "mlp_size": tune.qrandint(16, 32),
                "num_layers": 1,
                "generator_lr": tune.loguniform(1e-5, 1e-2),
                "discriminator_lr": tune.loguniform(1e-4, 1e-1),
                "init_mult1": tune.loguniform(1e-2, 1),
                "init_mult2": tune.loguniform(1e-2, 1),
                "weight_decay": 0.01
            }
    }
    
    # TuneConfig
    tune_config = tune.TuneConfig(
        metric='wasserstein_1d',
        mode='min',
        #search_alg=BayesOptSearch(metric="wasserstein_1d", mode="min"),  # Bayesian optimization
        search_alg=HyperOptSearch(metric="wasserstein_1d", mode="min"),
        scheduler=ASHAScheduler(),  # Early stopping
        #num_samples=5,  # Number of different hyperparameter combinations
        #max_concurrent_trials=5,  # Maximum concurrent trials
    )
    
    # RunConfig
    run_config = train.RunConfig(
        name='hyperparameter_tuning_SUD',
        local_dir='/Users/nicolafraccarolo/Documents/GitHub/SDE-Net/hyperparameter_tuning/ray_results',
        stop={
            "training_iteration": 10,
            "time_total_s": 20
        },
        verbose=2
    )
    
    # Trainable
    trainable = tune.with_resources(MyTrainableClass, resources={"cpu": 4})
    #trainable = MyTrainableClass
    
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
    


    
