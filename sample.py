import sys, os
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat

from model import Unet
from process import sig, Process 
from plot import save_vis, make_gif
from utils import save_path

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
        self.process = Process(args) # for sde_f and sde_g

        # shapes for image vs text datasets
        if args.dataset in ['mnist', 'cifar10']:
            self.end_shape = (32, 32)
        elif args.dataset == 'text8':
            self.end_shape = (8)

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.theta)
        sample = (np.sqrt(var) * torch.randn(self.batch_size, self.k-1, *self.end_shape)).to(self.device)
        x = sig(sample) 
        return x

    # TODO: re-impliment RK2
    # backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dB
    def update_order(self, model, x, t, dt):
        # get f, g, g^2 score and dB
        f = self.process.sde_f(x)
        g = self.process.sde_g(x)
        g2_score = model(x, t)

        # check f is not nan
        assert torch.isnan(f).sum() == 0, f'f is nan: {f}'
        dB = (0.1*np.sqrt(dt) * torch.randn_like(x)).to(self.device) 

        # solve sde: https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method 
        gdB = torch.einsum('b i j ..., b j ... -> b i ...', g, dB)
        return (f - g2_score)*dt #+ gdB

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png', order=1, d=10, pad=1e-4):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # select times for nn (always 0-1)
        t = torch.tensor([1.]).to(self.device)
        # dt for sde solver
        t_norm = args.t_max - args.t_min
        dt = t_norm * 1/T

        # noise schedule
        #g_scale = np.linspace(0,1,T)[::-1]
        #g_scale = 0.1*g_scale**2

        # sample loop
        for i in tqdm(range(T)):
            # update x
            change = self.update_order(model, x, t, dt)
            x += change
            t -= dt / t_norm

            # keep in simplex 
            x = torch.clamp(x, pad, 1-pad)
            xsum = x.sum(1, keepdim=True)
            x = torch.where(xsum > 1-pad, (1-pad)*x/xsum, x)

            # save sample
            if save_path is not None:
                # normalize change to be [0, 1] to visualize 
                change = (change - change.min()) / (change.max() - change.min())
                save_vis([x.clone(), change], f'imgs/{int(i/d)}.png', k=self.k)

        # discretize
        x = x.argmax(1)
        for i in range(int(T/d), int(T/d)+10):
            save_vis(x, f'imgs/{i}.png', k=self.k, a=self.a)

        # save gif
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
    sample_args = get_sample_args()
    path = os.path.join('results', sample_args.exp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    model = Unet(dim=64, channels=args.k-1).to(device)
    model_path = os.path.join('results', sample_args.exp, f'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    print(f'Loaded model from {model_path}')

    # sample from model
    sampler = Sampler(args, batch_size=8, device=device)
    sampler(model, T=1000, save_path=save_path(args, 'sample'))
