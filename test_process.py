import os, sys
import torch
import argparse
import numpy as np
from einops import repeat
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import get_args
from plot import make_gif

from utils import onehot2cat, a_logit
from plot import save_vis
from dataloader import mnist_dataset

# things that we need
# for a general process in d-1
# determine: O, h, a, [t_min, t_max]

# TODO: these transformations are biased!!
# generalized sigmoid
def sig(y, a):
    x_ = a * y.exp() / (1 + y.exp().sum(dim=1, keepdim=True))
    return x_

# generalized inverse sigmoid
def sig_inv(x, a):
    xd = a - x.sum(dim=1, keepdim=True)
    x_ = x[:, :-1]
    y = (x_/xd).log()
    return y

# score or grad log pdf
def score(x, mu=0, v=1, a=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # x_ -> x
    xd = a - x.sum(dim=1, keepdim=True)
    x = torch.cat((x, xd), dim=1)

    # n = d-1
    n = x.shape[1] - 1

    # set mu to tensor
    mu = mu * torch.ones(x.shape[0], n)

    # sig_a_inv of normal
    x_ = x[:, :-1]
    xd = x[:, -1].unsqueeze(1)

    # constant factor
    c1 = 1/(xd.squeeze()) * (x_.log() - xd.log() - mu).sum(dim=1)
    c1 = repeat(c1, 'b -> b k', k=n)

    # unique element-wise factor
    c2 = 1/(x_) * (x_.log() - xd.log() - mu)

    # change of variables component
    c3 = (x_ - xd) / (x_ * xd)

    score = -1/v * (c1 + c2) + c3
    return score

# step 1: get simplex size given d, pad g and dist q (rename)
def get_af(d, g=3, q=9):
    q = q**2
    # 1. cushion g on all sides on simplex
    # 2. length q from state at t_min to t_max
    r = np.sqrt(q * (d-1)) / np.sqrt(d)
    a1 = (-d*g - r + g) / (1/d - 1)
    a2 = (-d*g + r + g) / (1/d - 1)

    # take smallest possible value 
    m = min(a1, a2)
    if (m > 0 and m - d*g > 0):
        a = m
    else:
        a = max(a1, a2)
    
    # set value and return
    f = a - d*g
    return a, f

# step 2: find h, p_a(x) ~= sig( N(a/2, 1/(2h) ) as t -> inf
def get_h(d, a, h_init=1., N=1000, eps=1e-2, epochs=5000):
    # we want the logit-normal to match that of the normal at t -> inf
    # no closed for exists, so we use gradient descent
    h = torch.tensor(h_init, requires_grad=True)

    # optimizer for h
    opt = torch.optim.Adam([h], lr=1e-1)

    # run until convergence
    for i in range(epochs):
        # OU process at t -> inf, map to simplex
        y = torch.randn(N, d-1) / (2*h).sqrt()
        x = sig(y, a)

        # make variance 1
        var = x.var(dim=0)
        loss = (var - 1).pow(2).mean()

        # update h
        opt.zero_grad()
        loss.backward()
        opt.step()

        # for exit
        delta = loss.item()
        if delta < eps:
            return h.detach()

    # throw error if not converged
    raise ValueError(f'h did not converge after {epochs} iters')

if __name__ == '__main__':
    d = 2; pad = 3; dist = 9

    # step 1: get a :and x0
    a, f = get_af(d, g=pad, q=dist)
    print(f'a: {a:.5f}, x0: [{f + pad:.5f}, {pad}, ...]')

    # set initial value
    x0 = torch.zeros(d) + pad
    x0[0] += f

    # set final value
    x1 = torch.zeros(d) + a/d

    # check that sum is close
    #print(f'||x0||: {x0.sum().item():.5f} - should be {a:.5f}')
    #print(f'||x1||: {x1.sum().item():.5f} - should be {a:.5f}')
    #print(f'd(x1 - x0): {(x0 - x1).pow(2).sum().sqrt().item():.5f} - should be {dist:.5f}')

    # step 2: get h
    h = get_h(d, a)
    print(f'h: {h:.5f}')

    # check distribution of x
    y = torch.randn(10000, d-1) / (2*h).sqrt()
    x = sig(y, a)

    # plot distribution
    # use seaborn semi-transparent
    plt.hist(x[:, 0].detach().numpy(), bins=100, density=True, alpha=0.5)
    plt.hist((2*h).sqrt() * y[:, 0].detach().numpy() + a/2, bins=100, density=True, alpha=0.5)
    plt.legend(['x', 'y'])
    plt.show()
