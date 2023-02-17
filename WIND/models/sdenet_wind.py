import torch
import torch.nn as nn
# import torch.nn.functional as F
# import random
import torch.nn.init as init
import math

__all__ = ['SDENet_wind']


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ConcatConv1d(nn.Module):
    
    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv1d, self).__init__()
        module = nn.ConvTranspose1d if transpose else nn.Conv1d
        self._layer = module(   #???
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias
        )

    def forward(self, t, x):    #???
        tt = torch.ones_like(x[:, :1, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)
    

class Flatten(nn.Module):
    
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class Drift(nn.Module):
    
    def __init__(self, dim):
        super(Drift, self).__init__()
        self.norm1 = norm(dim=dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim_in=dim, dim_out=dim, ksize=3, stride=1, padding=1)
        self.norm2 = norm(dim=dim)
        self.conv2 = ConcatConv1d(dim_in=dim, dim_out=dim, ksize=3, stride=1, padding=1)
        self.norm3 = norm(dim=dim)

    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out   


class Diffusion(nn.Module):
    
    def __init__(self, dim_in, dim_out):
        super(Diffusion, self).__init__()
        self.norm1 = norm(dim=dim_in)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv1d(dim_in=dim_in, dim_out=dim_out, ksize=3, stride=1, padding=1)
        self.norm2 = norm(dim_in)
        self.conv2 = ConcatConv1d(dim_in=dim_in, dim_out=dim_out, ksize=3, stride=1, padding=1)
        self.fc = nn.Sequential(
            norm(dim_out), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(), 
            nn.Linear(in_features=dim_out, out_features=1), 
            nn.Sigmoid()
        )
        
    def forward(self, t, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.fc(out)
        return out
    

class SDENet_wind(nn.Module):
    
    def __init__(self, layer_depth, dim=64):
        super(SDENet_wind, self).__init__()
        self.layer_depth = layer_depth
        self.downsampling_layers = nn.Sequential(
            #                                                                                                            [N,C,H]
            #                                                                                                            [128,1,28]
            nn.Conv1d(in_channels=1, out_channels=dim, kernel_size=3, stride=1, padding=0, padding_mode='replicate'),   #[128,dim,26]
            norm(dim=dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), #[128,dim,13]
            norm(dim=dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, padding_mode='replicate'), #[128,dim,6]
        )
        self.drift = Drift(dim=dim)
        self.diffusion = Diffusion(dim_in=dim, dim_out=dim)
        self.fc_layers = nn.Sequential(
            norm(dim=dim), 
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool1d(output_size=1), 
            Flatten(), 
            nn.Linear(in_features=dim, out_features=1)
        )
        self.deltat = 6./self.layer_depth   #? why 6./self.layer_depth
        self.apply(init_params)
        self.sigma = 500
    
    def forward(self, x, training_diffusion=False):
        out = self.downsampling_layers(x)
        if not training_diffusion:
            t = 0
            diffusion_term = self.sigma*self.diffusion(t, out)
            diffusion_term = torch.unsqueeze(input=diffusion_term, dim=2)
            # diffusion_term = torch.unsqueeze(input=diffusion_term, dim=3)
            for i in range(self.layer_depth):
                t = 6*(float(i))/self.layer_depth
                #*
                #*
                #* STOCHASTIC DIFFERENTIAL EQUATION
                #* (Euler-Maruyama)
                out = out + self.drift(t, out)*self.deltat + diffusion_term*math.sqrt(self.deltat)*torch.randn_like(out).to(x)
            final_out = self.fc_layers(out)
        else:
            t = 0
            final_out = self.diffusion(t, out.detach())
        return final_out


def test():
    model = SDENet_wind(layer_depth=10, dim=64)
    return model
 
 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = test()
    num_params = count_parameters(model)
    print(num_params)
