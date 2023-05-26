import torch
import torch.nn as nn
import torch.nn.functional as F
# import random
import torch.nn.init as init
import math
from pathlib import Path
import json
from utils.log_utils import init_log

__all__ = ['SDENet_wind']


# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv1d):
#             init.kaiming_normal_(m.weight, mode='fan_out')
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm1d):
#             init.constant_(m.weight, 1)
#             init.constant_(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal_(m.weight, std=1e-3)
#             if m.bias is not None:
#                 init.constant_(m.bias, 0)


class Drift(nn.Module):
    def __init__(self):
        super(Drift, self).__init__()
        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, t, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.relu(out)
        return out


class Diffusion(nn.Module):
    def __init__(self):
        super(Diffusion, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(50, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 1)
        
    def forward(self, t, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        return out
    

class SDENet_wind(nn.Module):
    def __init__(self, layer_depth, H, zone=None, params=None, log_name=None):
        super(SDENet_wind, self).__init__()
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Linear(H, 50)
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(50, 1)
        )
        self.deltat = 4./self.layer_depth
        # self.apply(init_params)
        self.sigma = 5
        
        if log_name is None:
            self.log_name = 'sde-net_model'
            init_log(self.log_name, log_file=None, mode='w+')
        else:
            self.log_name = log_name
    
    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma * self.diffusion(t, out)
            for i in range(self.layer_depth):
                t = 4 * float(i) / self.layer_depth
                #*
                #* STOCHASTIC DIFFERENTIAL EQUATION
                #* (Euler-Maruyama)
                out = out \
                    + self.drift(t,out) * self.deltat \
                        + diffusion_term * math.sqrt(self.deltat) * torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
            final_out = final_out.squeeze(dim=1)
            return final_out


def test():
    model = SDENet_wind(layer_depth=10, H=64)
    return model
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)
