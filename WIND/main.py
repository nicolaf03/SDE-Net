import torch
import torch.backends.cudnn as cudnn

import argparse
import random
import data_loader

parser = argparse.ArgumentParser(description='SDE TCNN Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate of drift net')
parser.add_argument('--lr2', default=0.01, type=float, help='learning rate of diffusion net')
parser.add_argument('--training_out', action='store_false', default=True, help='training_with_out')
parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--eva_iter', default=5, type=int, help='number of passes when evaluation')
parser.add_argument('--zone', default='SUD', help='zone')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training')
# parser.add_argument('--imageSize', type=int, default=28, help='the height / width of the input image to network')
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=float, default=0)
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default=[10, 20,30], nargs='+', help='decreasing strategy')
parser.add_argument('--decreasing_lr2', default=[15, 30], nargs='+', help='decreasing strategy')
args = parser.parse_args()

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

torch.manual_seed(args.seed)
random.seed(args.seed)

if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    
print('load data:', args.zone)
train_loader_inDomain, test_loader_inDomain = data_loader.getDataSet(args.zone, args.batch_size, args.test_batch_size)
