import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from pathlib import Path
import time
from datetime import timedelta
import argparse
import random
import os
import sys
import path
import pdb
import math
import numpy as np
import json

from data_loader import data_loader
import models
from models import SDENet_wind
from utils.log_utils import init_log, dispose_log

import wandb
os.environ['WANDB_MODE'] = 'online'


def get_params(name, folder='parameters'):
    params_filename = 'params_' + name + '.json'
    with open(Path(folder) / params_filename) as json_file:
        params = json.load(json_file)
    return params


def train(parameters=None, plot=True, zone='mock'):
    
    curr_dir = Path(__file__).parent
    folder = curr_dir / '..' / 'trained_models'

    if not os.path.exists(folder):
        print(f'creating directory: {folder}')
        os.mkdir(folder)
        
    log_name = 'train_sde-net_model'
    log = init_log(log_name, curr_dir / '..' / 'logs' / (parameters + '_train.log'))
    log.info(f'parameters = {parameters}')
    start_time = time.time()
    
    log.info('loading model...')
    model = SDENet_wind.load_params(curr_dir / '..'  / 'parameters', parameters, log_name)
    
    log.info('loading training data...')
    model.load_training_data(curr_dir / '..' / 'data')
    
    log.info('train...')
    model.train(start_train, end_train, test_month, valid_month, plot=plot)
    
    log.info(f'elapsed time: {str(timedelta(seconds=time.time() - start_time))}')
    

    #------------------------------------------------------------------------------------------
    
    parser = argparse.ArgumentParser(description='SDE-Net Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate of drift net')
    parser.add_argument('--lr2', default=0.01, type=float, help='learning rate of diffusion net')
    # parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
    parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
    #
    parser.add_argument('--zone', default='mock', help='zone')
    parser.add_argument('--h', default=1, help='time horizon forecasting')
    parser.add_argument('--H', default=100, help='length of history')
    #
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=float, default=0)
    parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--decreasing_lr', default=[20], nargs='+', help='decreasing strategy')
    parser.add_argument('--decreasing_lr2', default=[], nargs='+', help='decreasing strategy')
    
    args = parser.parse_args()
    print(args)

    # check if gpu is available
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if device == 'cuda':
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        
    
    print('load data:', args.zone)
    train_loader, test_loader = data_loader.getDataSet(args.zone, args.H, args.h, args.batch_size, args.test_batch_size)

    # Model
    print('==> Building model..')
    net = models.SDENet_wind(layer_depth=4, H=args.H)
    net = net.to(device)


    real_label = 0
    fake_label = 1

    # criterion = nn.L1Loss()     
    # criterion = nn.GaussianNLLLoss()
    criterion = nn.BCELoss()
    
    def nll_loss(y, mean, sigma):
        loss = torch.mean(torch.log(sigma) + (y - mean)**2 / (sigma**2))
        return loss
    def mse(y_pred, y_true):
        loss = torch.mean((y_pred - y_true)**2)
        return loss
    # def mape(y_pred, y_true):
    #     loss = torch.mean(torch.abs(y_pred - y_true) / y_true)
    #     return loss
    # mse = nn.MSELoss()

    optimizer_F = optim.SGD(
        params=[
            {'params': net.downsampling_layers.parameters()}, 
            {'params': net.drift.parameters()}, 
            {'params': net.fc_layers.parameters()},
            {'params': net.diffusion.parameters()}
        ],
        lr=args.lr,
        #momentum=0.4,
        weight_decay=5e-4
    )
    optimizer_G = optim.SGD(
        params=[{'params': net.diffusion.parameters()}],
        lr=args.lr2,
        momentum=0.9,
        weight_decay=5e-4
    )

    # optimizer_F = optim.AdamW(
    #     params=[{'params': net.downsampling_layers.parameters()}, {'params': net.drift.parameters()}, {'params': net.fc_layers.parameters()}],
    #     lr=0.001
    # )
    # optimizer_G = optim.AdamW(
    #     params=[{'params': net.diffusion.parameters()}],
    #     lr=0.001
    # )
    # use a smaller sigma during training for training stability
    # net.sigma = 20

    # training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        
        if epoch == 0:
            net.sigma = 0.1
        if epoch == 30:
            net.sigma = 0.5

        train_loss = 0
        # train_loss_in = 0
        # train_loss_out = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            #
            # training with in-domain data
            optimizer_F.zero_grad()
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            mu, sigma = net(inputs, training_diffusion=False)
            # predict = net(inputs, training_diffusion=False)
            
            loss = nll_loss(targets, mu, sigma)
            # loss = mse(predict, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 100.)
            train_loss += loss.item()
            
            optimizer_F.step()
            
            # #
            # # training with out-of-domain data
            # optimizer_G.zero_grad()
        
            # # tensor full of zeros
            # label = torch.full((args.batch_size,1), float(real_label), device=device)
            # predict_in = net(inputs, training_diffusion=True)
            # #
            # # distance between non-noisy data and 0
            # # binary cross entropy because there are only 2 classes: 0 (no noise) and 1 (maximum noise)
            # loss_in = criterion(predict_in, label)
            # loss_in.backward()
            # train_loss_in += loss_in.item()
            
            # # tensor full of ones
            # label.fill_(fake_label)
            # inputs_out = inputs + 2 * torch.randn(args.batch_size, 1, args.H, device=device)
            # predict_out = net(inputs_out, training_diffusion=True)
            # #
            # # distance between noisy data and 1
            # # binary cross entropy because there are only 2 classes: 0 (no noise) and 1 (maximum noise)
            # loss_out = criterion(predict_out, label)
            # loss_out.backward()
            # train_loss_out += loss_out.item()
            
            # optimizer_G.step()
            

        print('Train epoch: {} \tNLL: {:.6f}'
        .format(epoch, train_loss/(len(train_loader))))
        # print('Train epoch: {} \tLoss: {:.6f} | Loss_in: {:.6f} | Loss_out: {:.6f}'
        # .format(epoch, train_loss/(len(train_loader)), train_loss_in/(len(train_loader)), train_loss_out/(len(train_loader))))

        return train_loss / len(train_loader)

    def test(epoch):
        net.eval()
        test_loss_mse = 0
        test_loss_nll = 0
        with torch.no_grad():
            deltat = 1
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)

                current_mu = 0
                current_sigma = 0
                for i in range(args.eva_iter):
                    mu, sigma = net(inputs)
                    current_mu = current_mu + mu
                    current_sigma = current_sigma + sigma
                current_mu = current_mu / args.eva_iter
                current_sigma = current_sigma / args.eva_iter
                #*
                #*
                #* Euler-Maruyama
                x_in = inputs[:,:,-1]
                x_out = x_in \
                    + current_mu * deltat \
                        + current_sigma * math.sqrt(deltat) * torch.randn_like(x_in)
                
                loss_mse = mse(targets, x_out)
                test_loss_mse += loss_mse.item()
                loss_nll = nll_loss(targets, current_mu, current_sigma)
                test_loss_nll += loss_nll.item()

            print(' Test epoch: {} \tNLL: {:.6f} \tMSE: {:.6f}'
            .format(epoch, test_loss_nll/len(test_loader), test_loss_mse/len(test_loader)))
            
        return test_loss_nll / len(test_loader)


    wandb.init(project='wind_neural')
    for epoch in range(0, args.epochs):
        train_loss = train(epoch)
        wandb.log({'train loss': train_loss})
        test_loss = test(epoch)
        wandb.log({'test loss': test_loss})
        if epoch in args.decreasing_lr:
            for param_group in optimizer_F.param_groups:
                param_group['lr'] *= args.droprate
        if epoch in args.decreasing_lr2:
            for param_group in optimizer_G.param_groups:
                param_group['lr'] *= args.droprate

    print('Saving model...')
    if not os.path.isdir('./WIND/trained_model'):
        os.makedirs('./WIND/trained_model')
    torch.save(net.state_dict(), f'./WIND/trained_model/model_{args.zone}')


if __name__ == '__main__':
    
    train()
    # curr_dir = Path(__file__).parent
    # params = get_params(folder=curr_dir / '..'  / 'parameters', name='CSUD_v11')
    print(0)