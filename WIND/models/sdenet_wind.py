import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
import logging
import math
from pathlib import Path
import json
import logging

from WIND.utils.log_utils import init_log

__all__ = ['SDENet_wind']
logger = logging.getLogger(__name__)
from ..utils.log_utils import init_log
from .fBM import hosking



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
    def __init__(self, layer_depth, H, Hurst_idx, zone=None, params=None, log_name=None):
        super(SDENet_wind, self).__init__()
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Linear(H, 50)
        self.drift = Drift()
        self.diffusion = Diffusion()
        self.fc_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(50, 2)
        )
        self.deltat = 4./self.layer_depth
        # self.apply(init_params)
        self.sigma = 5
        self.sde_solution = self.set_sde_sol('abm', T=H, N=H, H=Hurst_idx)
        
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
                out = self.sde_solution(t, out, diffusion_term)
            final_out = self.fc_layers(out)
            mu = final_out[:,:,0]
            sigma = F.softplus(final_out[:,:,1]) + 1e-3
            return mu, sigma
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
            sigma = final_out[:, :, 0]
            return sigma

    def set_sde_sol(self, form: str, T, N, H):
        """
        The corrent function defines the form of the SDE.
        Currently 2 different SDEs are implemented (Geometric Brownian Motion) and Arithmentic.
        Returns:
        """
        fBM = hosking(T, N, H)
        match form:
            case 'abm':
                sde_sol = lambda t, out, diffusion_term: out \
                    + self.drift(t, out) * self.deltat \
                        + diffusion_term * fBM[t]
            case 'gbm':
                sde_sol = lambda t, out, diffusion_term: out * (1 + self.drift(t, out) * self.deltat + diffusion_term * fBM[t])

            case _:
                logger.info('SDE form has been chosen by default')

                sde_sol = lambda t, out, diffusion_term: out + self.drift(t, out) * self.deltat \
                                                             + diffusion_term * fBM[t]
        return sde_sol

    @staticmethod
    def load_params(folder, name, log_name=None):
        params_filename = 'params_' + name + '.json'
        with open(Path(folder) / params_filename) as json_file:
            params = json.load(json_file)

        return SDENet_wind(params=params, log_name=log_name)

    def set_sde_form(self, form: str):
        """
        The corrent function defines the form of the SDE.
        Currently 2 different SDEs are implemented (Geometric Brownian Motion) and Arithmentic.
        Returns:

        """
        match form:
            case 'abm':
                sde_form = lambda t, out, x, diffusion_term: out \
                    + self.drift(t, out) * self.deltat \
                        + diffusion_term * math.sqrt(self.deltat) * torch.randn_like(out).to(x)
            case 'gbm':
                sde_form = lambda t, out, x, diffusion_term: out * (1 + self.drift(t, out) * self.deltat \
                                                       + diffusion_term * math.sqrt(
                                                        self.deltat) * torch.randn_like(out).to(x))
            case _:
                logger.info('SDE form has been chosen by default')
                sde_form = lambda t, out, x, diffusion_term: out \
                                                             + self.drift(t, out) * self.deltat \
                                                             + diffusion_term * math.sqrt(
                    self.deltat) * torch.randn_like(out).to(x)
        return sde_form

def test():
    model = SDENet_wind(layer_depth=10, H=64)
    return model
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)
