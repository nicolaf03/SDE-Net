from model.gan_model import GanModel
import utils.constants as constants
from utils.log_utils import init_log, dispose_log

from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from pathlib import Path
import os
import time

curr_dir = Path(__file__).parent
log_name = 'prediction'
log = init_log(log_name, curr_dir / '..' / 'logs' / f'{log_name}.log', mode='a')


def load_model(models_folder, parameters, model_name, device):
    model = GanModel.load_params(curr_dir / '..' / 'parameters', parameters)
    log.info(f'loading model: {model_name}')
    model.load_model(models_folder, model_name, device)
    log.info(f'model train_params: {model.train_params}')
    return model


def predict(zone, version, test_window, device):
    curr_dir = Path(__file__).parent
    models_folder = curr_dir / '..' / 'trained_models'
    
    log.info(f'zone: {zone}')
    parameters = f'{zone}_{version}'
    model_name = f'{zone}_{version}'
    
    model = load_model(models_folder, parameters, model_name, device)
    
    log.info('loading training data...')
    model.load_training_data(curr_dir / '..' / 'data')
    
    start_test, end_test = test_window
    log.info(f'predict from {start_test} to {end_test} ...')
    generated_samples, y_test, mean_err = model.predict(test_window, device, plot=True)
    print(mean_err)
    log.info('done prediction!')

    dispose_log(log)
    

if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    #device = 'cuda' if is_cuda else 'cpu'
    device = 'cpu'
    
    zone = 'SUD'
    version = 'v5'
    test_window = ('2021-07', '2021-12')
    predict(zone, version, test_window, device)