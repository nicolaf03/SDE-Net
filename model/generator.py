import numpy as np
import torch
import torchcde
import torchsde
from model.mlp import MLP
from fractional_BM.fBM import FractionalBM

class GeneratorFunc(torch.nn.Module):
    sde_type = 'stratonovich'
    noise_type = 'general'

    def __init__(self, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._noise_size = noise_size
        self._hidden_size = hidden_size

        ###################
        # Drift and diffusion are MLPs. They happen to be the same size.
        # Note the final tanh nonlinearity: this is typically important for good performance, to constrain the rate of
        # change of the hidden state.
        # If you have problems with very high drift/diffusions then consider scaling these so that they squash to e.g.
        # [-3, 3] rather than [-1, 1].
        ###################
        self._drift = MLP(1 + hidden_size, hidden_size, mlp_size, num_layers, tanh=True)
        self._diffusion = MLP(1 + hidden_size, hidden_size * noise_size, mlp_size, num_layers, tanh=True)

    def f_and_g(self, t, x):
        # t has shape ()
        # x has shape (batch_size, hidden_size)
        t = t.expand(x.size(0), 1)
        tx = torch.cat([t, x], dim=1)
        #return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)
        #!! cambiato
        return torch.zeros_like(self._drift(tx)), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


class Generator(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        data_size = 1
        try:
            H = params['H']
        except:
            H = 0.5
        initial_noise_size = params['initial_noise_size']
        noise_size = params['noise_size']
        hidden_size = params['hidden_size']
        mlp_size = params['mlp_size']
        num_layers = params['num_layers']
        
        self.h = H
        
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)
        # self._bm_h = self.get_fractional_noise(torch.tensor([0., 1., 2., 3., 4., 5., 6.]), n_sims)
    
    
    def get_fractional_trajectories(self, ts, n_sims, _bm_h):
        all_idx = np.arange(0, n_sims)
        idx = np.random.choice(all_idx, size=n_sims, replace=False)
        return torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], H=_bm_h[idx])
    
    
    def get_fractional_noise(self, ts, n_sims):
        # We generate the Fractional Noise
        h = 0.5
        fBM = FractionalBM()
        fBM.set_parameters([1, 0, 0, 1, h])
        t_steps = 1 # note it can be modified
        fBM_noise = torch.from_numpy(fBM.simulate(n_sims=n_sims, t_steps=t_steps, dt=1/(t_steps * (ts.size(0)))).values)[:, -1:] # we are interested in the noise at T
        # fBM_noise1 = torch.tensor(fBM_noise, dtype=torch.float32)
        fBM_noise0 = fBM_noise.clone().detach().type(torch.float32)
        # fBM.simulate(n_sims=self.n_sims, t_steps=t_steps, dt=1 / (t_steps * (ts.size(0)))).values)[:, -1:]
        return fBM_noise0

    
    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)
        
        h = self.h
        if h == 0.5:
            bm_h = None
        else:
            n_sims = x0.size(0)
            _bm_h = self.get_fractional_noise(torch.tensor([0., 1., 2., 3., 4., 5., 6.]), n_sims)
            bm_h = self.get_fractional_trajectories(ts, n_sims, _bm_h)
        
        #****************************************
        # We generate the Fractional Noise
        # h = 0.5
        # fBM = FractionalBM()
        # fBM.set_parameters([1, 0, 0, 1, h])
        # t_steps = 1
        # fBM_noise = torch.from_numpy(fBM.simulate(n_sims=x0.size(0), t_steps=t_steps, dt=1/(t_steps * (ts.size(0)))).values)[:, -1:] # we are interested in the noise at T
        
        # fBM_noise = fBM_noise.to(ts.device)
        # bm_h = torchsde.BrownianInterval(t0=ts[0], t1=ts[-1], H=fBM_noise.float())
        #****************************************
        
        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(
            self._func, x0, ts, 
            method='reversible_heun', 
            adjoint_method='adjoint_reversible_heun',
            dt=1.0,
            #**********
            bm=bm_h
            #**********
        )
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))

  
if __name__ == '__main__':
    print('hello')
    
    params = {
        "custom": {
            "name": "SUD_model_v2",
            "zone": "SUD",
            "t_size": 7,
            "batch_size": 16,
            "n_epochs": 10000,
            "patience": 1000,
            "steps": 200,
            "swa_step_start": 5000,
            "steps_per_print": 10,
            "num_plot_samples": 50,
            "plot_locs": [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        "gan": {
            "initial_noise_size": 4,
            "noise_size": 4,
            "hidden_size": 14,
            "mlp_size": 88,
            "num_layers": 1,
            "generator_lr": 2e-4,
            "discriminator_lr": 1e-3,
            "init_mult1": 3,
            "init_mult2": 0.5,
            "weight_decay": 0.01
        }
    }
    
    batch_size = params['custom']['batch_size']
    t_size = params['custom']['t_size']
    ts = torch.linspace(0, t_size - 1, t_size)
    
    generator = Generator(params['gan'])
    generator(ts, batch_size)
        