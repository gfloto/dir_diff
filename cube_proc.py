import os, sys
import torch
import argparse
import numpy as np
from einops import repeat
import matplotlib.pyplot as plt

'''
diffusion on cube sigmoid(OU)
'''

class CubeProcess:
    def __init__(self, args):
        self.O = args.O
        self.t_min = args.t_min
        self.t_max = args.t_max
        self.theta = args.theta

        self.k = args.k # number of categories (before log_2)
        self.device = args.device

        # useful attributes
        if args.dataset in ['mnist', 'cifar10']:
            self.data_type = 'image'

        elif args.dataset in ['text8']:
            self.data_type = 'text'
            self.char2idx = {char: idx for idx, char in enumerate(
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '])}
            self.idx2char = {idx: char for char, idx in self.char2idx.items()}

    # get t, rescale to be in proper interval
    def t(self):
        tu = torch.rand(1)
        t = (self.t_max - self.t_min) * tu + self.t_min
        return t.to(self.device), tu.to(self.device)

    # mean and variance of OU process at time t
    # true is a product identical up to sign...
    def xt(self, x0, t):
        x0 = 2*x0 - 1

        # get mean and variance of OU process
        mu = (-self.theta * t).exp() * self.O * x0
        var = 1/(2*self.theta) * (1 - (-2*self.theta * t).exp())

        # sample in R, then map to simplex
        sample = var.sqrt() * torch.randn_like(x0) + mu
        xt = torch.sigmoid(sample)
        return xt, mu, var

    # compute score g^2 score, which is used for numerical stability
    def g2_score(self, xt, mu, var):
        return - (xt*(xt-1)) * (2*xt + mu/var - torch.logit(xt)/var - 1)
    
    # make drift term sde
    def sde_f(self, xt):
        z = xt * (1-xt)
        return self.theta*torch.logit(xt)*z + 0.5*z*(1-2*xt)

    # make diffusion term sde
    def sde_g(self, xt):
        return xt * (1-xt)

'''
testing scripts for process and time sampler
'''

from tqdm import tqdm
from args import get_args
from plot import make_gif, save_vis
from dataloader import text8_dataset, mnist_dataset, cifar10_dataset

if __name__ == '__main__':
    N = 20; chars = 50

    # get device, data and process
    args = get_args()
    if args.dataset == 'text8':
        loader = text8_dataset(args)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args)
    elif args.dataset == 'cifar10':
        loader = cifar10_dataset(args)
    process = CubeProcess(args)

    # test forward process
    t = torch.linspace(args.t_min, args.t_max, N).to(args.device)

    # get x0
    x0 = next(iter(loader))
    if isinstance(x0, tuple) or isinstance(x0, list):
        x0 = x0[0] 
    x0 = x0.to(args.device)
   
    # print initial text
    if process.data_type == 'text':
        print(f't: {0:.3f}, text: {process.decode_text(x0)[0][:chars]}')

    # get forward process at each t
    os.makedirs('imgs', exist_ok=True)
    for i in range(N):
        # generate and save image
        xt, _, _ = process.xt(x0.clone(), t[i])
        print(xt[0,:,0,0,0])

        if process.data_type == 'image':
            save_vis(xt, f'imgs/{i}.png', k=args.k)
        else:
            print(f't: {t[i]:.3f}, text: {process.decode_text(xt)[0][:chars]}')

    # make gif of forward process
    if process.data_type == 'image':
        make_gif('imgs', f'results/forward_simplex_{args.dataset}.gif', N)


