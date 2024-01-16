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


def load_model(models_folder, parameters, model_name, device):
    model = GanModel.load_params(curr_dir / '..' / 'parameters', parameters)
    model.load_model(models_folder, model_name, device)
    return model


def predict_month(zone, version, start_test, end_test, device):
    curr_dir = Path(__file__).parent
    models_folder = curr_dir / '..' / 'trained_models'
    
    parameters = f'{zone}_{version}'
    model_name = f'{zone}_{version}'
    
    model = load_model(models_folder, parameters, model_name, device)
    
    model.load_training_data(curr_dir / '..' / 'data')
    
    model.custom_params['t_size'] = 31
    generated_samples, real_samples = model.predict2(start_test, end_test, device, plot=False)


if __name__ == '__main__':
    device = 'cpu'    
    zone = 'SUD'
    version = 'v1'
    
    start_test = '2020-01-01'
    end_test = '2020-01-31'
    predict_month(zone, version, start_test, end_test, device)