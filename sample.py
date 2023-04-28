import sys, os
import torch
import numpy as np
from tqdm import tqdm

from utils import a_logit, save_path
from model import Unet
from plot import save_vis, make_gif

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv_a(x) x(1-x/a) + 0.5 x(1-x/a)(1-2x/a)
g(x) = x(1-x/a)
'''

class Sampler:
    def __init__(self, args, batch_size, device):
        self.O = torch.tensor(args.O)
        self.h = torch.tensor(args.h)
        self.a = torch.tensor(args.a)
        self.t_min, self.t_max = args.T
        self.device = device

        self.batch_size = batch_size
        self.img_shape = (32, 32)

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*self.t_max))
        nu = torch.randn(self.batch_size, *self.img_shape)
        x = self.a * torch.sigmoid(var.sqrt() * nu)
        return x

    # drift term
    def f(self, x):
        a = self.a
        f1 = -self.h*a_logit(x,a) * x*(1 - x/a)
        f2 = 0.5*x*(1 - x/a) * (1 - 2*x/a)
        return f1 + f2 

    # diffusion term
    def g(self, x):
        return x*(1-x/self.a)

    # score at xt given mu and var
    def s(self, xt, mu, var):
        num = a_logit(xt, self.a) - 2*var*xt - mu + self.a*var
        denom = var*xt*(xt - self.a)
        score = num / denom
        return score
    
    def update_order(self, model, x, t, dt, order=1, g_scale=0.02):
        # get info for euler discretization of sde solution
        f = self.f(x)
        g = self.g(x)
        eps = torch.randn(self.batch_size, *self.img_shape).to(self.device) 
        score = model(x[:, None, ...], t).squeeze()

        # runge kutta solvers
        if order == 1:
            update = (f + g**2 * score)*dt + g_scale*g*eps 
        elif order == 2:
            k1 = dt * (f + g**2 * score) # k1 is same as euler first order
            x1 = x + k1
            f1 = self.f(x1)
            g1 = self.g(x1)
            t1 = t - dt

            score1 = model(x1[:, None, ...], t1).squeeze()
            k2 = dt * (f1 + g1**2 * score1)
            update = (k1+k2)/2 + g_scale*g1*eps 
        return update
            

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png', order=1, d=10):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # select times for nn (always 0-1)
        dt = torch.tensor([1/T]).to(self.device)
        t = torch.tensor([1.]).to(self.device)

        # noise schedule
        g_scale = np.linspace(0,1,T)[::-1]
        g_scale = 0.1*g_scale**2

        # sample loop
        for i in tqdm(range(T)):
            update = self.update_order(model, x, t, dt, order=order, g_scale=g_scale[i])

            # update x
            x += update
            t -= dt

            # save sample
            if save_path is not None:
                x_save = x / self.a # map back to prob. simplex
                save_vis(x_save, f'imgs/{int(i/d)}.png', k=None, x_out=update)

        # binarize
        x = x / self.a # map back to prob. simplex
        x = (x > 0.5).float()
        for i in range(int(T/d), int(T/d)+10):
            save_vis(x, f'imgs/{i}.png', k=None)

        # save gif
        if save_path is not None:
            make_gif('imgs', save_path, int(T/d)+10)

import json
import argparse
if __name__ == '__main__':
    name = 'dev'
    path = os.path.join('results', name)

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    model = Unet(dim=64, channels=1).to('cuda')
    model_path = os.path.join('results', name, f'model_{args.proc_name}.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    # print model name
    # sample from model
    sampler = Sampler(args, batch_size=8, device='cuda')
    sampler(model, T=1000, save_path=save_path(args, 'sample'))
