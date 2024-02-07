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


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    print(figs)
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    plt.close()
    pp.close()


def train(device, train_window, valid_window, test_window, parameters, plot=True, zone='', add_folder=None):
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
    log.info('loading training data...')
    model.load_training_data(curr_dir / '..' / 'data')
    log.info('train...')
    print(f'begin train at {start_time}')
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

    train(
        device=device,
        train_window=('2015-01','2020-12'),
        valid_window=('2021-01','2021-09'),
        test_window=('2021-10','2021-12'),
        parameters='SUD_v3',
        zone='SUD',
        add_folder=None
    )
    
    
    
    # parameters = 'SUD_v1'
    # add_folder = None
    # curr_dir = Path(__file__).parent
    # folder = curr_dir / '..' / 'trained_models'
    # if add_folder != None:
    #     folder = folder / add_folder

    # if not os.path.exists(folder):
    #     print(f"creating directory: {folder}")
    #     os.mkdir(folder)

    # log_name = 'train_gan_model'
    # log = init_log(log_name, curr_dir / '..' / 'logs' / (parameters + '_train.log'))
    # log.info(f'parameters = {parameters}')
    # start_time = time.time()
    # log.info('loading model...')
    # model = GanModel.load_params(curr_dir / '..'  / 'parameters', parameters, log_name)