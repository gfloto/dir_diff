import os, sys
import torch
import argparse
import numpy as np
from einops import repeat
import matplotlib.pyplot as plt

from plot import save_vis
from dataloader import mnist_dataset

# TODO: these transformations are biased!!
def sig(y):
    x_ = y.exp() / (1 + y.exp().sum(dim=1, keepdim=True))
    return x_

# generalized inverse sigmoid
def sig_inv(x_, a):
    xd = 1 - x_.sum(dim=1, keepdim=True)
    y = (x_).log() - (xd).log()
    return y

'''
main diffusion process class
'''

class Process:
    def __init__(self, args):
        self.O = args.O
        self.t_min = args.t_min
        self.t_max = args.t_max
        self.theta = torch.tensor(args.theta).to(args.device)

        self.k = args.k
        self.device = args.device

    # get t, rescale to be in proper interval
    def t(self):
        tu = torch.rand(1)
        t = (self.t_max - self.t_min) * tu + self.t_min
        return t.to(self.device), tu.to(self.device)

    # mean and variance of OU process at time t
    def xt(self, x0, t):
        # convert from S -> O
        O_ = x0[:, :-1]
        O_ *= self.O # this is one-hot * O

        # make [-O, -O, ...] is cat is kth class
        ksum = repeat(O_.sum(dim=1), 'b ... -> b k ...', k=self.k-1)
        O = torch.where(ksum == 0, -self.O, O_)

        # get mean and variance of OU process
        mu = (-self.theta * t).exp() * O
        var = 1/(2*self.theta) * (1 - (-2*self.theta * t).exp())

        # sample in R, then map to simplex
        sample = var.sqrt() * torch.randn_like(O) + mu
        xt = sig(sample)
        return xt, mu, var

    # compute score
    def score(self, x_, mu, var):
        # get last component
        xd = 1 - x_.sum(dim=1, keepdim=True)

        # constant factor
        c1 = 1/(xd.squeeze()) * (x_.log() - xd.log() - mu).sum(dim=1)
        c1 = repeat(c1, 'b ... -> b k ...', k=self.k-1)

        # unique element-wise factor
        c2 = 1/(x_) * (x_.log() - xd.log() - mu)

        # change of variables component
        c3 = (x_ - xd) / (x_ * xd)

        score = -1/var * (c1 + c2) + c3
        return score

'''
testing scripts for process and time sampler
'''

from tqdm import tqdm
from utils import get_args
from plot import make_gif

if __name__ == '__main__':
    N = 100 # number of steps

    # get device, data and process
    args = get_args()
    loader = mnist_dataset(8, args.k)
    process = Process(args)

    # test forward process
    t = torch.linspace(*args.T, N).to(args.device)
    (x0, _) = next(iter(loader))
    x0 = x0.to(args.device)
    out = process.O(x0)

    print('running forward process...')
    os.makedirs('imgs', exist_ok=True)

    # get forward process at each t
    for i in tqdm(range(N)):
        # generate and save image
        xt = process.xt(x0, t[i])
        save_vis(xt, f'imgs/{i}.png', args.a, args.k)

    # make gif of forward process
    make_gif('imgs', 'results/forward.gif', N)


