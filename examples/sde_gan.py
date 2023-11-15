# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""#*Train an SDE as a GAN, on data from a time-dependent Ornstein--Uhlenbeck process.

Training SDEs as GANs was introduced in "Neural SDEs as Infinite-Dimensional GANs".
https://arxiv.org/abs/2102.03657

#*This reproduces the toy example in Section 4.1 of that paper.

This additionally uses the improvements introduced in "Efficient and Accurate Gradients for Neural SDEs".
https://arxiv.org/abs/2105.13493

To run this file, first run the following to install extra requirements:
pip install fire
pip install git+https://github.com/patrick-kidger/torchcde.git

To run, execute:
python -m examples.sde_gan
"""
from pathlib import Path
import fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim.swa_utils as swa_utils
import torchcde
import torchsde
import tqdm
import os
import argparse

import wandb
os.environ['WANDB_MODE'] = 'offline'

curr_dir = Path(__file__).parent

###################
# First some standard helper objects.
###################

class LipSwish(torch.nn.Module):
    def forward(self, x):
        return 0.909 * torch.nn.functional.silu(x)


class MLP(torch.nn.Module):
    def __init__(self, in_size, out_size, mlp_size, num_layers, tanh):
        super().__init__()

        model = [torch.nn.Linear(in_size, mlp_size),
                 LipSwish()]
        for _ in range(num_layers - 1):
            model.append(torch.nn.Linear(mlp_size, mlp_size))
            ###################
            # LipSwish activations are useful to constrain the Lipschitz constant of the discriminator.
            # (For simplicity we additionally use them in the generator, but that's less important.)
            ###################
            model.append(LipSwish())
        model.append(torch.nn.Linear(mlp_size, out_size))
        if tanh:
            #model.append(torch.nn.Tanh())
            model.append(torch.nn.Sigmoid())
        self._model = torch.nn.Sequential(*model)

    def forward(self, x):
        return self._model(x)


###################
# Now we define the SDEs.
#
# We begin by defining the generator SDE.
###################
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
        return self._drift(tx), self._diffusion(tx).view(x.size(0), self._hidden_size, self._noise_size)


###################
# Now we wrap it up into something that computes the SDE.
###################
class Generator(torch.nn.Module):
    def __init__(self, data_size, initial_noise_size, noise_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._initial_noise_size = initial_noise_size
        self._hidden_size = hidden_size

        self._initial = MLP(initial_noise_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = GeneratorFunc(noise_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, data_size)

    def forward(self, ts, batch_size):
        # ts has shape (t_size,) and corresponds to the points we want to evaluate the SDE at.

        ###################
        # Actually solve the SDE.
        ###################
        init_noise = torch.randn(batch_size, self._initial_noise_size, device=ts.device)
        x0 = self._initial(init_noise)

        ###################
        # We use the reversible Heun method to get accurate gradients whilst using the adjoint method.
        ###################
        xs = torchsde.sdeint_adjoint(self._func, x0, ts, method='reversible_heun', dt=1.0,
                                     adjoint_method='adjoint_reversible_heun',)
        xs = xs.transpose(0, 1)
        ys = self._readout(xs)

        ###################
        # Normalise the data to the form that the discriminator expects, in particular including time as a channel.
        ###################
        ts = ts.unsqueeze(0).unsqueeze(-1).expand(batch_size, ts.size(0), 1)
        return torchcde.linear_interpolation_coeffs(torch.cat([ts, ys], dim=2))


###################
# Next the discriminator. Here, we're going to use a neural controlled differential equation (neural CDE) as the
# discriminator, just as in the "Neural SDEs as Infinite-Dimensional GANs" paper. (You could use other things as well,
# but this is a natural choice.)
#
# There's actually a few different (roughly equivalent) ways of making the discriminator work. The curious reader is
# encouraged to have a read of the comment at the bottom of this file for an in-depth explanation.
###################
class DiscriminatorFunc(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()
        self._data_size = data_size
        self._hidden_size = hidden_size

        # tanh is important for model performance
        self._module = MLP(1 + hidden_size, hidden_size * (1 + data_size), mlp_size, num_layers, tanh=True)

    def forward(self, t, h):
        # t has shape ()
        # h has shape (batch_size, hidden_size)
        t = t.expand(h.size(0), 1)
        th = torch.cat([t, h], dim=1)
        return self._module(th).view(h.size(0), self._hidden_size, 1 + self._data_size)


class Discriminator(torch.nn.Module):
    def __init__(self, data_size, hidden_size, mlp_size, num_layers):
        super().__init__()

        self._initial = MLP(1 + data_size, hidden_size, mlp_size, num_layers, tanh=False)
        self._func = DiscriminatorFunc(data_size, hidden_size, mlp_size, num_layers)
        self._readout = torch.nn.Linear(hidden_size, 1)

    def forward(self, ys_coeffs):
        # ys_coeffs has shape (batch_size, t_size, 1 + data_size)
        # The +1 corresponds to time. When solving CDEs, It turns out to be most natural to treat time as just another
        # channel: in particular this makes handling irregular data quite easy, when the times may be different between
        # different samples in the batch.

        Y = torchcde.LinearInterpolation(ys_coeffs)
        Y0 = Y.evaluate(Y.interval[0])
        h0 = self._initial(Y0)
        hs = torchcde.cdeint(Y, self._func, h0, Y.interval, method='reversible_heun', backend='torchsde', dt=1.0,
                             adjoint_method='adjoint_reversible_heun',
                             adjoint_params=(ys_coeffs,) + tuple(self._func.parameters()))
        score = self._readout(hs[:, -1])
        return score.mean()


###################
# Generate some data. For this example we generate some synthetic data from a time-dependent Ornstein-Uhlenbeck SDE.
###################
def _load_data(zone, H):
    PATH = curr_dir / '..' / 'data' / f'wind_{zone}_train.csv'
    data = pd.read_csv(PATH)
    value_array = np.array(data.iloc[:,1], dtype='float32')
    values = []
    
    for i in range(len(data)-H):
        sub_array = value_array[i:i+H]
        x = torch.from_numpy(np.expand_dims(sub_array,0))
        values.append(x)
    
    return torch.stack(values).transpose(1,2)


def get_data(zone, batch_size, plot_data=False):
    t_size = 64

    ts = torch.linspace(0, t_size - 1, t_size)      # [64]
    ys = _load_data(zone=zone, H=t_size)            # [2651, 64, 1]
    
    dataset_size = ys.shape[0]

    ###################
    # Typically important to normalise data. Note that the data is normalised with respect to the statistics of the
    # initial data, _not_ the whole time series. This seems to help the learning process, presumably because if the
    # initial condition is wrong then it's pretty hard to learn the rest of the SDE correctly.
    ###################
    y0_flat = ys[0].view(-1)
    y0_not_nan = y0_flat.masked_select(~torch.isnan(y0_flat)) #? unnecessary
    ys = (ys - y0_not_nan.mean()) / y0_not_nan.std()
    
    # todo: sistema senza ciclo
    # sto traslando tutte le traiettorie in modo che partano da 0
    for i in range(ys.size()[0]):
        ys[i] = ys[i] - ys[:,0,:][i]

    ###################
    # As discussed, time must be included as a channel for the discriminator.
    ###################
    ys = torch.cat([ts.unsqueeze(0).unsqueeze(-1).expand(dataset_size, t_size, 1), ys], dim=2)  # [2651, 64, 2]

    ###################
    # Package up
    ###################
    data_size = ys.size(-1) - 1  # How many channels the data has (not including time, hence the minus one).
    ys_coeffs = torchcde.linear_interpolation_coeffs(ys)  # as per neural CDEs.
    dataset = torch.utils.data.TensorDataset(ys_coeffs)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if plot_data:
        _plot(ts, dataloader, num_plot_samples=50, zone=zone)

    return ts, data_size, dataloader


def _plot(ts, dataloader, num_plot_samples, zone):
    if not os.path.isdir(f'./torchsde/images/{zone}'):
        os.makedirs(f'./torchsde/images/{zone}')
        
    # Get samples
    real_samples, = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_samples = real_samples[..., 1]

    real_samples = real_samples[:num_plot_samples]

    # Plot samples
    plt.figure()
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
    
    plt.title(f"{num_plot_samples} samples from both real distribution")
    plt.tight_layout()
    plt.savefig(f'./torchsde/images/{zone}/real_samples_{zone}.png', dpi=200, format='png')
    #plt.show()


###################
# We'll plot some results at the end.
###################
def plot(ts, generator, dataloader, num_plot_samples, plot_locs, zone):
    if not os.path.isdir(f'./torchsde/images/{zone}'):
        os.makedirs(f'./torchsde/images/{zone}')
        
    # Get samples
    real_samples, = next(iter(dataloader))
    assert num_plot_samples <= real_samples.size(0)
    real_samples = torchcde.LinearInterpolation(real_samples).evaluate(ts)
    real_samples = real_samples[..., 1]

    with torch.no_grad():
        generated_samples = generator(ts, real_samples.size(0)).cpu()
    generated_samples = torchcde.LinearInterpolation(generated_samples).evaluate(ts)
    generated_samples = generated_samples[..., 1]

    # Plot histograms
    for prop in plot_locs:
        time = int(prop * (real_samples.size(1) - 1))
        real_samples_time = real_samples[:, time]
        generated_samples_time = generated_samples[:, time]
        
        plt.figure()
        _, bins, _ = plt.hist(real_samples_time.cpu().numpy(), bins=32, alpha=0.7, label='Real', color='dodgerblue',
                              density=True)
        bin_width = bins[1] - bins[0]
        num_bins = int((generated_samples_time.max() - generated_samples_time.min()).item() // bin_width)
        plt.hist(generated_samples_time.cpu().numpy(), bins=num_bins, alpha=0.7, label='Generated', color='crimson',
                 density=True)
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title(f'Marginal distribution at time {time}.')
        plt.tight_layout()
        plt.savefig(f'./torchsde/images/{zone}/marginal_distribution_{prop}_{zone}.png', dpi=200, format='png')
        #plt.show()

    real_samples = real_samples[:num_plot_samples]
    generated_samples = generated_samples[:num_plot_samples]

    # Plot samples
    plt.figure()
    real_first = True
    generated_first = True
    for real_sample_ in real_samples:
        kwargs = {'label': 'Real'} if real_first else {}
        plt.plot(ts.cpu(), real_sample_.cpu(), color='dodgerblue', linewidth=0.5, alpha=0.7, **kwargs)
        real_first = False
    for generated_sample_ in generated_samples:
        kwargs = {'label': 'Generated'} if generated_first else {}
        plt.plot(ts.cpu(), generated_sample_.cpu(), color='crimson', linewidth=0.5, alpha=0.7, **kwargs)
        generated_first = False
    plt.legend()
    plt.title(f"{num_plot_samples} samples from both real and generated distributions.")
    plt.tight_layout()
    plt.savefig(f'./torchsde/images/{zone}/samples_real_vs_generated_{zone}.png', dpi=200, format='png')
    plt.show()


###################
# Now do normal GAN training, and plot the results.
#
# GANs are famously tricky and SDEs trained as GANs are no exception. Hopefully you can learn from our experience and
# get these working faster than we did -- we found that several tricks were often helpful to get this working in a
# reasonable fashion:
# - Stochastic weight averaging (average out the oscillations in GAN training).
# - Weight decay (reduce the oscillations in GAN training).
# - Final tanh nonlinearities in the architectures of the vector fields, as above. (To avoid the model blowing up.)
# - Adadelta (interestingly seems to be a lot better than either SGD or Adam).
# - Choosing a good learning rate (always important).
# - Scaling the weights at initialisation to be roughly the right size (chosen through empirical trial-and-error).
###################

def evaluate_loss(ts, batch_size, dataloader, generator, discriminator, device):
    with torch.no_grad():
        total_samples = 0
        total_loss = 0
        for real_samples, in dataloader:
            generated_samples = generator(ts, batch_size).to(device)
            generated_score = discriminator(generated_samples)
            real_score = discriminator(real_samples.to(device))
            loss = generated_score - real_score
            total_samples += batch_size
            total_loss += loss.item() * batch_size
    return total_loss / total_samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--zone', type=str, default='mock')
    #
    # Architectural hyperparameters. These are quite small for illustrative purposes.
    parser.add_argument('--initial_noise_size', type=int, default=5)
    parser.add_argument('--noise_size', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--mlp_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=1)
    #
    # Training hyperparameters. Be prepared to tune these very carefully, as with any GAN.
    parser.add_argument('--generator_lr', type=float, default=2e-4)
    parser.add_argument('--discriminator_lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--init_mult1', type=float, default=3)
    parser.add_argument('--init_mult2', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--swa_step_start', type=int, default=5000)
    #
    # Evaluation and plotting hyperparameters
    parser.add_argument('--steps_per_print', type=int, default=10)
    parser.add_argument('--num_plot_samples', type=int, default=50)
    parser.add_argument('--plot_locs', type=tuple, default=(0.1, 0.3, 0.5, 0.7, 0.9))
    #
    args = parser.parse_args()
    
    is_cuda = torch.cuda.is_available()
    device = 'cuda:0' if is_cuda else 'cpu'
    if not is_cuda:
        print("Warning: CUDA not available; falling back to CPU but this is likely to be very slow.")
    
    # Data
    print(f'Retrieve data for zone {args.zone}')
    ts, data_size, train_dataloader = get_data(zone=args.zone, batch_size=args.batch_size, plot_data=True)
    ts = ts.to(device)
    infinite_train_dataloader = (elem for it in iter(lambda: train_dataloader, None) for elem in it)

    # Models
    generator = Generator(data_size, args.initial_noise_size, args.noise_size, args.hidden_size, args.mlp_size, args.num_layers).to(device)
    discriminator = Discriminator(data_size, args.hidden_size, args.mlp_size, args.num_layers).to(device)
    # Weight averaging really helps with GAN training.
    averaged_generator = swa_utils.AveragedModel(generator)
    averaged_discriminator = swa_utils.AveragedModel(discriminator)

    # Picking a good initialisation is important!
    # In this case these were picked by making the parameters for the t=0 part of the generator be roughly the right
    # size that the untrained t=0 distribution has a similar variance to the t=0 data distribution.
    # Then the func parameters were adjusted so that the t>0 distribution looked like it had about the right variance.
    # What we're doing here is very crude -- one can definitely imagine smarter ways of doing things.
    # (e.g. pretraining the t=0 distribution)
    with torch.no_grad():
        for param in generator._initial.parameters():
            param *= args.init_mult1
        for param in generator._func.parameters():
            param *= args.init_mult2

    # Optimisers. Adadelta turns out to be a much better choice than SGD or Adam, interestingly.
    generator_optimiser = torch.optim.Adadelta(generator.parameters(), lr=args.generator_lr, weight_decay=args.weight_decay)
    discriminator_optimiser = torch.optim.Adadelta(discriminator.parameters(), lr=args.discriminator_lr,
                                                   weight_decay=args.weight_decay)

    # Train both generator and discriminator.
    trange = tqdm.tqdm(range(args.steps))
    wandb.init(project='wind_gan')
    for step in trange:
        real_samples, = next(infinite_train_dataloader)
        real_samples = real_samples.to(device)

        generated_samples = generator(ts, args.batch_size)
        generated_samples = generated_samples.to(device)
        
        generated_score = discriminator(generated_samples)
        generated_score = generated_score.to(device)
        
        real_score = discriminator(real_samples)
        real_score = real_score.to(device)
        
        loss = generated_score - real_score # todo: capire il senso di questa loss
        loss = loss.to(device)
        
        wandb.log({'train loss': loss})
        
        loss.backward()

        for param in generator.parameters():
            param.grad *= -1
        generator_optimiser.step()
        discriminator_optimiser.step()
        generator_optimiser.zero_grad()
        discriminator_optimiser.zero_grad()

        ###################
        # We constrain the Lipschitz constant of the discriminator using carefully-chosen clipping (and the use of
        # LipSwish activation functions).
        ###################
        with torch.no_grad():
            for module in discriminator.modules():
                if isinstance(module, torch.nn.Linear):
                    lim = 1 / module.out_features
                    module.weight.clamp_(-lim, lim)

        # Stochastic weight averaging typically improves performance.
        if step > args.swa_step_start:
            averaged_generator.update_parameters(generator)
            averaged_discriminator.update_parameters(discriminator)

        if (step % args.steps_per_print) == 0 or step == args.steps - 1:
            total_unaveraged_loss = evaluate_loss(ts, args.batch_size, train_dataloader, generator, discriminator, device=device)
            if step > args.swa_step_start:
                total_averaged_loss = evaluate_loss(ts, args.batch_size, train_dataloader, averaged_generator.module,
                                                    averaged_discriminator.module, device=device)
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f} "
                             f"Loss (averaged): {total_averaged_loss:.4f}")
            else:
                trange.write(f"Step: {step:3} Loss (unaveraged): {total_unaveraged_loss:.4f}")
    generator.load_state_dict(averaged_generator.module.state_dict())
    discriminator.load_state_dict(averaged_discriminator.module.state_dict())

    _, _, test_dataloader = get_data(zone=args.zone, batch_size=args.batch_size)
    
    print('Saving model...')
    if not os.path.isdir('./torchsde/trained_model'):
        os.makedirs('./torchsde/trained_model')
    torch.save(generator.state_dict(), f'./torchsde/trained_model/generator_{args.zone}')
    torch.save(discriminator.state_dict(), f'./torchsde/trained_model/discriminator_{args.zone}')

    plot(ts, generator, test_dataloader, args.num_plot_samples, args.plot_locs, args.zone)


###################
# And that's (one way of doing) an SDE as a GAN. Have fun.
###################

###################
# Appendix: discriminators for a neural SDE
#
# This is a little long, but should all be quite straightforward. By the end of this you should have a comprehensive
# knowledge of how these things fit together.
#
# Let Y be the real/generated sample, and let H be the hidden state of the discriminator.
# For real data, then Y is some interpolation of an (irregular) time series. (As with neural CDEs, if you're familiar -
# for a nice exposition on this see https://github.com/patrick-kidger/torchcde/blob/master/example/irregular_data.py.)
# In the case of generated data, then Y is _either_ the continuous-time sample produced by sdeint, _or_ it is an
# interpolation (probably linear interpolation) of the generated sample between particular evaluation points, We'll
# refer to these as cases (*) and (**) respectively.
#
# In terms of the mathematics, our options for the discriminator are:
# (a1) Solve dH(t) = f(t, H(t)) dt + g(t, H(t)) dY(t),
# (a2) Solve dH(t) = (f, g)(t, H(t)) d(t, Y(t))
# (b) Solve dH(t) = f(t, H(t), Y(t)) dt.
# Option (a1) is what is stated in the paper "Neural SDE as Infinite-Dimensional GANs".
# Option (a2) is theoretically the same as (a1), but the drift and diffusion have been merged into a single function,
# and the sample Y has been augmented with time. This can sometimes be a more helpful way to think about things.
# Option (b) is a special case of the first two, by Appendix C of arXiv:2005.08926.
# [Note that just dH(t) = g(t, H(t)) dY(t) would _not_ be enough, by what's known as the tree-like equivalence property.
#  It's a bit technical, but the basic idea is that the discriminator wouldn't be able to tell how fast we traverse Y.
#  This is a really easy mistake to make; make sure you don't fall into it.]
#
# Whether we use (*) or (**), and (a1) or (a2) or (b), doesn't really affect the quality of the discriminator, as far as
# we know. However, these distinctions do affect how we solve them in terms of code. Depending on each combination, our
# options are to use a solver of the following types:
#
#      | (a1)   (a2)   (b)
# -----+----------------------
#  (*) | SDE           SDE
# (**) |        CDE    ODE
#
# So, (*) implies using an SDE solver: the continuous-time sample is only really available inside sdeint, so if we're
# going to use the continuous-time sample then we need to solve generator and discriminator together inside a single SDE
# solve. In this case, as our generator takes the form
# Y(t) = l(X(t)) with dX(t) = μ(t, X(t)) dt + σ(t, X(t)) dW(t),
# then
# dY(t) = l(X(t)) dX(t) = l(X(t))μ(t, X(t)) dt + l(X(t))σ(t, X(t)) dW(t).
# Then for (a1) we get
# dH(t) = ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) ) dt + g(t, H(t))l(X(t))σ(t, X(t)) dW(t),
# which we can now put together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t)                                )      ( σ(t, X(t))                  )
# d( Y(t) ) = ( l(X(t))μ(t, X(t)                         ) dt + ( l(X(t))σ(t, X(t))           ) dW(t)
#  ( H(t) )   ( f(t, H(t)) + g(t, H(t))l(X(t))μ(t, X(t)) )      ( g(t, H(t))l(X(t))σ(t, X(t)) ),
# whilst for (b) we can put things together into one big SDE solve:
#  ( X(t) )   ( μ(t, X(t))       )      ( σ(t, X(t))        )
# d( Y(t) ) = ( l(X(t))μ(t, X(t) ) dt + ( l(X(t))σ(t, X(t)) ) dW(t)
#  ( H(t) )   ( f(t, H(t), Y(t)) )      ( 0                 )
#
# Phew, what a lot of stuff to write down. Don't be put off by this: there's no complicated algebra, it's literally just
# substituting one equation into another. Also, note that all of this is for the _generated_ data. If using real data,
# then Y(t) is as previously described always an interpolation of the data. If you're able to evaluate the derivative of
# the interpolation then you can then apply (a1) by rewriting it as dY(t) = (dY/dt)(t) dt and substituting in. If you're
# able to evaluate the interpolation itself then you can apply (b) directly.
#
# The benefit of using (*) is that everything can be done inside a single SDE solve, which is important if you're
# thinking about using adjoint methods and the like, for memory efficiency. The downside is that the code gets a bit
# more complicated: you need to be able to solve just the generator on its own (to produce samples at inference time),
# just the discriminator on its own (to evaluate the discriminator on the real data), and the combined
# generator-discriminator system (to evaluate the discriminator on the generated data).
#
# Right, let's move on to (**). In comparison, this is much simpler. We don't need to substitute in anything. We're just
# taking our generated data, sampling it at a bunch of points, and then doing some kind of interpolation (probably
# linear interpolation). Then we either solve (a2) directly with a CDE solver (regardless of whether we're using real or
# generated data), or solve (b) directly with an ODE solver (regardless of whether we're using real or generated data).
#
# The benefit of this is that it's much simpler to code: unlike (*) we can separate the generator and discriminator, and
# don't ever need to combine them. Also, real and generated data is treated the same in the discriminator. (Which is
# arguably a good thing anyway.) The downside is that we can't really take advantage of things like adjoint methods to
# backpropagate efficiently through the generator, because we need to produce (and thus store) our generated sample at
# lots of time points, which reduces the memory efficiency.
#
# Note that the use of ODE solvers for (**) is only valid because we're using _interpolated_ real or generated data,
# and we're assuming that we're using some kind of interpolation that is at least piecewise smooth. (For example, linear
# interpolation is piecewise smooth.) It wouldn't make sense to apply ODE solvers to some rough signal like Brownian
# motion - that's what case (*) and SDE solvers are about.
#
# Right, let's wrap up this wall of text. Here, we use option (**), (a2). This is arguably the simplest option, and
# is chosen as we'd like to keep the code readable in this example. To solve the CDEs we use the CDE solvers available
# through torchcde: https://github.com/patrick-kidger/torchcde.
###################