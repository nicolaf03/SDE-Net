from model.gan_model import GanModel
from utils.log_utils import init_log, dispose_log

from datetime import timedelta
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from pathlib import Path
import os
import time

import argparse

parser = argparse.ArgumentParser('SDE GAN')
parser.add_argument('--version', type=str, default='v3')
parser.add_argument('--version_old', default=None)
parser.add_argument('--wandb', type=str, default='offline')
args = parser.parse_args()

import wandb
os.environ['WANDB_MODE'] = args.wandb
os.environ['WANDB__SERVICE_WAIT'] = '30'


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    print(figs)
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    plt.close()
    pp.close()


def train(
    device, 
    train_window, 
    valid_window, 
    test_window, 
    parameters, 
    plot=True, 
    zone='', 
    pre_trained_model=None, 
    add_folder=None
):
    curr_dir = Path(__file__).parent
    folder = curr_dir / '..' / 'trained_models'
    if add_folder != None:
        folder = folder / add_folder

    if not os.path.exists(folder):
        print(f"creating directory: {folder}")
        os.mkdir(folder)

    log_name = 'train_gan_model'
    log = init_log(log_name, curr_dir / '..' / 'logs' / (parameters + '_train.log'))
    log.info(f'parameters = {parameters}')
    start_time = time.time()
    
    log.info('loading model...')
    model = GanModel.load_params(curr_dir / '..'  / 'parameters', parameters, log_name)
    
    if pre_trained_model is not None:
        log.info('loading pre-trained model...')
        print(f'loading pre-trained model: {pre_trained_model}')
        model.load_model(curr_dir / '..' / 'trained_models', pre_trained_model, device)
        model.custom_params['patience'] = model.custom_params['patience'] / 2
    
    log.info('loading training data...')
    model.load_training_data(curr_dir / '..' / 'data')
    log.info('train...')
    model.train(device, train_window, test_window, valid_window, plot=plot)
    
    log.info(f'elapsed time: {str(timedelta(seconds=time.time() - start_time))}')
    model_name = f'{parameters}'
    log.info(f'save model: {model_name}')
    files = model.save_model(folder, model_name)
    
    #for file in files:
        #s3_client.upload_file(Path(folder) / file, f'models/pod/new_models/{file}', overwrite=False)

    dispose_log(log)
    if plot:
        multipage(f'{folder}/_stats_models_{zone}.pdf')



if __name__ == "__main__":
    
    # todo: si puo' spostare device dentro alla funzione train di GanModel
    is_cuda = torch.cuda.is_available()
    device = 'cuda' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    else:
        print("Using cuda")
    
    #zones = constants.ZONE
    zone = 'SUD'
    params = f'{zone}_{args.version}'
    if args.version_old is not None:
        params_old = f'{zone}_{args.version_old}'
    else:
        params_old = None
    
    train(
        device=device,
        train_window=('2015-01','2020-12'),
        valid_window=('2021-01','2021-09'),
        test_window=('2021-10','2021-12'),
        parameters=params,
        zone=zone,
        pre_trained_model=params_old,
        add_folder=None
    )