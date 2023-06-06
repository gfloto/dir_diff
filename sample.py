import sys, os
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from model import Unet
from process import Process, sig 
from cube_proc import CubeProcess
from plot import save_vis, make_gif
from utils import save_path

# get entropy on first dim
def entropy(x):
    xd = 1 - torch.sum(x, dim=1, keepdim=True)
    x = torch.cat((x, xd), dim=1)
    print(x[:,:,15,15])
    return -(x * x.log()).sum(dim=1)

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv_a(x) x(1-x/a) + 0.5 x(1-x/a)(1-2x/a)
g(x) = x(1-x/a)
'''

class Sampler:
    def __init__(self, args, batch_size, device):
        self.O = args.O
        self.t_min = args.t_min
        self.t_max = args.t_max
        self.theta = args.theta 

        self.k = args.k
        self.device = device
        self.batch_size = batch_size
        self.proc_type = args.proc_type
        
        # process object for sde f and g
        if args.proc_type == 'simplex':
            self.process = Process(args)
        elif args.proc_type == 'cube':
            self.process = CubeProcess(args)

        # shapes for image vs text datasets
        if args.dataset in ['mnist', 'cifar10']:
            self.end_shape = (32, 32)
        elif args.dataset == 'text8':
            self.end_shape = (8)
        elif args.dataset == 'city':
            self.end_shape = (32, 64)

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.theta)
        k_ = args.k-1 if self.proc_type == 'simplex' else args.k
        if args.dataset in ['mnist', 'city']:
            sample = (np.sqrt(var) * torch.randn(self.batch_size, k_, *self.end_shape)).to(self.device)
        elif args.dataset == 'cifar10':
            sample = (np.sqrt(var) * torch.randn(self.batch_size, k_, 3, *self.end_shape)).to(self.device)
        x = sig(sample) 
        return x

    # backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dB
    # TODO: re-impliment RK2
    def update(self, model, x, t, dt, g_scale=1):
        # get f, g, g^2 score and dB
        f = self.process.sde_f(x)
        g = self.process.sde_g(x)     

        # set t to tensor, then get score
        t = torch.tensor([t]).float().to(self.device)
        g2_score = model(x, t)

        # check f is not nan
        assert torch.isnan(f).sum() == 0, f'f is nan: {f}'
        dB = (np.sqrt(dt) * torch.randn_like(g2_score)).to(self.device) 

        # solve sde: https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method 
        if self.proc_type == 'simplex':
            gdB = torch.einsum('b i j ..., b j ... -> b i ...', g, dB)
        elif self.proc_type == 'cube':
            gdB = g * dB
        return (-f + g2_score)*dt + g_scale*gdB

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png', order=1, d=10, pad=1e-2, vis=True):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # noise schedule
        g_scale = np.linspace(0,1,T)[::-1]
        g_scale = 1.75*np.power(g_scale, 1.5)

        # time schedule
        t = 1.
        dt = (self.t_max - self.t_min) / T

        # sample loop
        d = 20
        for i in tqdm(range(T)):
            # update x
            change = self.update(model, x, t, dt)#, g_scale[i])
            x = x + change
            t -= 1/T

            # keep in simplex 
            if self.proc_type == 'simplex':
                x = torch.clamp(x, pad, 1-pad)
                xsum = x.sum(1, keepdim=True)
                x = torch.where(xsum > 1-pad, (1-pad)*x/xsum, x)

            # keep in cube
            elif self.proc_type == 'cube':
                x = torch.clamp(x, pad, 1-pad)

            # save sample
            # normalize change to be [0, 1] to visualize 
            if save_path is not None and vis and i%d == 0:
                save_vis([x.clone(), entropy(x.clone())], f'imgs/{int(i/d)}.png', k=self.k, simplex=[True, False])

        # discretize
        if vis:
            for i in range(int(T/d), int(T/d)+10):
                simplex = self.proc_type == 'simplex'
                save_vis([x.clone(), entropy(x.clone())], f'imgs/{i}.png', k=self.k, simplex=[True, False])

        # save gif
        if vis:
            if save_path is not None:
                make_gif('imgs', save_path, int(T/d)+10)

import json
import argparse

# to load experiment
def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, help='experiment name')
    args = parser.parse_args()

    assert args.exp is not None, 'Must specify experiment name'
    return args

if __name__ == '__main__':
    batch_size = 256
    sample_args = get_sample_args()
    path = os.path.join('results', sample_args.exp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    if args.dataset in ['mnist', 'city']:
        ch = args.k if args.proc_type in ['cat', 'cube'] else args.k-1
        model = Unet(dim=64, channels=ch).to(args.device)
    elif args.dataset == 'cifar10':
        ch = 3*args.k if args.proc_type in ['cat', 'cube'] else 3*(args.k-1)
        model = Unet(dim=128, channels=ch).to(args.device)

    model_path = os.path.join('results', sample_args.exp, f'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    print(f'Loaded model from {model_path}')

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    # sample from model
    sampler = Sampler(args, batch_size=batch_size, device=device)
    sampler(model, T=1000, save_path=save_path(args, 'sample.gif'))
