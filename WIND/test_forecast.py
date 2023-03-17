###################################################################################################
# Measure the forecast performance
###################################################################################################

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
# import calculate_log as callog
import models
import math
import os 
import random
from torchvision import datasets, transforms
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from numpy.linalg import inv
import numpy as np


def main():
    
    # Training settings
    parser = argparse.ArgumentParser(description='Test code - measure the detection peformance')
    parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
    # parser.add_argument('--network', type=str, choices=['resnet', 'sdenet','mc_dropout'], default='resnet')
    parser.add_argument('--network', type=str, default='mock')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--seed', type=float, default=0, help='random seed')
    #
    parser.add_argument('--zone', default='mock', help='zone')
    parser.add_argument('--h', default=1, help='time horizon forecasting')
    parser.add_argument('--H', default=100, help='length of history')
    #
    parser.add_argument('--out_dataset', required=False, help='out-of-dist dataset: cifar10 | svhn | imagenet | lsun')
    parser.add_argument('--pre_trained_net', default='./save_sdenet_wind/final_model', help="path to pre trained_net")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--test_batch_size', type=int, default=1)

    args = parser.parse_args()
    print(args)

    outf = 'test/' + args.network
    if not os.path.isdir(outf):
        os.makedirs(outf)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # print("Random Seed: ", args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)


    print('Load model')
    model = models.SDENet_wind(layer_depth=4, H=args.H)

    model.load_state_dict(torch.load(args.pre_trained_net))
    model = model.to(device)
    model_dict = model.state_dict()

    print('load target data: ', args.zone)
    _, test_loader = data_loader.getDataSet(args.zone, args.H, args.h, args.batch_size, args.test_batch_size)

    #print('load non target data: ',args.out_dataset)
    #nt_train_loader, nt_test_loader = data_loader.getDataSet(args.out_dataset, args.batch_size, args.test_batch_size, args.imageSize)


    def mse(y_pred, y_true):
            loss = torch.mean((y_pred - y_true)**2)
            return loss


    def generate_target():
        model.eval()  
        test_loss = 0
        total = 0
        f1 = open('%s/confidence_Base_In.txt'%outf, 'w')

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                mu, sigma = model(inputs)
                
                current_mu = 0
                for i in range(args.eva_iter):
                    mu, sigma = model(inputs)
                    current_mu = mu + current_mu
                    if i == 0:
                        Sigma = torch.unsqueeze(sigma,1)
                        Mean = torch.unsqueeze(mu,1)
                    else:
                        Sigma = torch.cat((Sigma, torch.unsqueeze(sigma,1)), dim=1)
                        Mean = torch.cat((Mean, torch.unsqueeze(mu,1)), dim=1)
                        
                current_mu = current_mu / args.eva_iter
                loss = mse(targets, current_mu)
                test_loss += loss.item()
                
                Var_mean = Mean.std(dim=1)
                for i in range(inputs.size(0)):
                    soft_out = Var_mean[i].item()
                    f1.write("{}\n".format(-soft_out))

        f1.close()

        print('\nFinal RMSE: {}'
        .format(np.sqrt(test_loss/len(test_loader))))



    print('generate log from in-distribution data')
    generate_target()
    # print('generate log  from out-of-distribution data')
    # generate_non_target()
    # print('calculate metrics for OOD')
    # callog.metric(outf, 'OOD')
    # print('calculate metrics for mis')
    # callog.metric(outf, 'mis')


if __name__ == '__main__':
    
    main()
