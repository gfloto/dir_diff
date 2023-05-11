import os, sys
import torch
import argparse
import numpy as np
from einops import repeat
import matplotlib.pyplot as plt

from utils import identity_tensor

# TODO: these transformations are biased!!
# map from R^k -> simplex
def sig(y):
    x_ = y.exp() / (1 + y.exp().sum(dim=1, keepdim=True))
    return x_

# map from simplex -> R^k
def sig_inv(x_):
    xd = 1 - x_.sum(dim=1, keepdim=True)
    y = x_.log() - xd.log()
    return y

'''
main diffusion process class
'''

# TODO: make all simplex variables s (not x)
class Process:
    def __init__(self, args):
        self.O = args.O
        self.t_min = args.t_min
        self.t_max = args.t_max
        self.theta = args.theta

        self.k = args.k
        self.s_dist = None
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
        # sample t from uniform distribution
        if self.s_dist is None:
            tu = torch.rand(1)
            t = (self.t_max - self.t_min) * tu + self.t_min
            return t.to(self.device), tu.to(self.device)
        # sample from s_dist which is categorical 
        # then cample uniformly from that interval
        else:
            # get bin and region (on [0,1])
            bins = len(self.s_dist)
            bin_id = np.random.choice(np.arange(bins), p=self.s_dist)
            bin_min = bin_id / bins
            bin_max = (bin_id + 1) / bins

            # get tu and t
            tu_ = torch.rand(1)
            tu = tu_ * (bin_max - bin_min) + bin_min
            t = (self.t_max - self.t_min) * tu + self.t_min
            return t.to(self.device), tu.to(self.device)

    # mean and variance of OU process at time t
    def xt(self, x0, t):
        assert x0.shape[1] == self.k

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

    # g^2 * score
    def g2_score(self, xt, mu, var):
        # useful function
        score = self.score(xt, mu, var)

        g = self.sde_g(xt)
        g2 = torch.einsum('b i j ..., b j k ... -> b i k ...', g, g)
        g2_score = torch.einsum('b i j ..., b j ... -> b i ...', g2, score)
        
        return g2_score

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
    
    # make drift term sde
    def sde_f(self, s):
        b, k = s.shape[:2]

        x = sig_inv(s)
        beta = -self.theta*x + 0.5*(1-2*s)

        # \sum_{i \neq j} X_{ij}, vectorized
        beta_v = repeat(s*beta, 'b d ... -> b k d ...', k=self.k-1)
        I = identity_tensor(s)
        bsum = torch.einsum('b i j ..., b i j ... -> b i ...', 1-I, beta_v)

        f = s * ( (1-s)*beta - bsum )
        return f

    # make diffusion term sde
    def sde_g(self, s):
        b, k = s.shape[:2]
        I = identity_tensor(s)

        neq = torch.einsum('b i ..., b j ... -> b i j ...', s, s)
        eq = repeat(s*(1-s), 'b d ... -> b k d ...', k=self.k-1)

        g1 = torch.einsum('b i j ..., b i j ... -> b i j ...', I, eq)
        g2 = torch.einsum('b i j ..., b i j ... -> b i j ...', 1-I, neq)

        g = g1 - g2
        return g

    # one-hot text to string
        # add last component
    def decode_text(self, encoded_text):
        xd = 1 - torch.sum(encoded_text, dim=1, keepdim=True)
        encoded_text = torch.cat((encoded_text, xd), dim=1)

        # Convert one-hot encoding to indices
        indices = torch.argmax(encoded_text, dim=1)

        # Convert indices to characters using the idx2char dictionary
        decoded_text = [''.join([self.idx2char[idx.item()]
                                for idx in example]) for example in indices]
        return decoded_text 

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
        loader = text8_dataset(args.batch_size)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args.batch_size, args.k)
    elif args.dataset == 'cifar10':
        loader = cifar10_dataset(args.batch_size, args.k)
    process = Process(args)

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

        if process.data_type == 'image':
            save_vis(xt, f'imgs/{i}.png', k=args.k)
        else:
            print(f't: {t[i]:.3f}, text: {process.decode_text(xt)[0][:chars]}')

    # make gif of forward process
    if process.data_type == 'image':
        make_gif('imgs', f'results/forward_simplex_{args.dataset}.gif', N)
