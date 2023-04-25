import os, sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import onehot2cat, a_logit
from plot import save_vis
from dataloader import mnist_dataset

'''
main diffusion process class
'''

class Process:
    def __init__(self, args):
        #self.O = torch.tensor(args.O).to(args.device)
        self.h = torch.tensor(args.h).to(args.device)
        self.a = torch.tensor(args.a).to(args.device)
        self.t_min, self.t_max = args.T
        self.device = args.device

    # get t, rescale to be in proper interval
    def t(self):
        tu = torch.rand(1)
        t = (self.t_max - self.t_min) * tu + self.t_min
        return t.to(self.device), tu.to(self.device)

    # mean and variance of OU process at time t
    def mean_var(self, x0, t):
        mu = self.O * torch.exp(-self.h*t) * x0
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*t)) * torch.ones_like(x0).to(x0.device)
        return mu, var

    # sample from logit normal distribution: R -> S
    def sample(self, mu, var):
        return self.a * torch.sigmoid(mu + var.sqrt()*torch.randn_like(mu))

    # return xt given x0 and t
    def xt(self, x0, t):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance, then sample
        mu, var = self.mean_var(x0, t)
        xt = self.sample(mu, var)
        return xt

    # compute score at xt given mu and var
    def s(self, xt, mu, var):
        a = self.a
        num = a*a_logit(xt, a) - 2*var*xt - a*mu + a*var
        denom = var*xt*(xt - a)
        score = num / denom
        return score

    # score at xt given x0 and t
    def score(self, x0, xt, t):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance
        mu, var = self.mean_var(x0, t)
        score = self.s(xt, mu, var)
        return score


'''
testing scripts for process and time sampler
'''

from tqdm import tqdm
from utils import get_args
from plot import make_gif

if __name__ == '__main__':
    N = 5 # number of steps

    # get device, data and process
    args = get_args()
    loader = mnist_dataset(8, args.k)
    process = Process(args)

    # test forward process
    t = torch.linspace(*args.T, N).to(args.device)
    (x0, _) = next(iter(loader))
    x0 = x0.to(args.device)

    print('running forward process...')
    os.makedirs('imgs', exist_ok=True)

    # get forward process at each t
    for i in tqdm(range(N)):
        # generate and save image
        xt = process.xt(x0, t[i])
        save_vis(xt, f'imgs/{i}.png', args.a, args.k)

    # make gif of forward process
    make_gif('imgs', 'results/forward.gif', N)


