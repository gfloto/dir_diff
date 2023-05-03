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

# TODO: this code is a nightmare!! clean it up

# TODO: these transformations are biased!!
# generalized sigmoid
def sig(y):
    x_ = y.exp() / (1 + y.exp().sum(dim=1, keepdim=True))
    return x_

# generalized inverse sigmoid
def sig_inv(x_):
    xd = 1 - x_.sum(dim=1, keepdim=True)
    y = x._log() - xd.log()
    return y

# score or grad log pdf
def score(x, mu=0, v=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # x_ -> x
    xd = 1 - x.sum(dim=1, keepdim=True)
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

# find theta s.t. 3std is close to desired value as t -> inf 
def get_theta(k, theta_init=5., sig3=0.1, N=1000, eps=1e-4, epochs=5000):
    theta = torch.tensor(theta_init, requires_grad=True)

    # optimizer for h
    opt = torch.optim.Adam([theta], lr=1e-1)

    # run until convergence
    for i in range(epochs):
        # OU process at t -> inf, map to simplex
        X = torch.randn(N, k-1) / (2*theta).sqrt()
        S = sig(X)

        # make variance 1
        std = 3 * S.std(dim=0)
        loss = (std - sig3).pow(2).mean()

        # update h
        opt.zero_grad()
        loss.backward()
        opt.step()

        # for exit
        delta = loss.item()
        if delta < eps:
            return theta.detach()

    # throw error if not converged
    raise ValueError(f'h did not converge after {epochs} iters')

# one hot in first position
def fhot(O, d):
    # one hot
    onht = torch.zeros(d-1)
    onht[0] = 1
    return O * onht

# helpful function to sample from OU process
def sample(O, theta, t, k, N=1000):
    # get mean and variance of OU process
    mu = O*(-theta*t).exp()
    var = 1/(2*theta) * (1 - (-2*theta*t).exp())

    # sample in R, then map to simplex
    X = var.sqrt() * torch.randn(N, k-1) + mu
    S = sig(X)
    return S

# get t_min
def get_Ot(s0, theta, k, O_init=3., t_init=0.1, N=1000, epochs=5000):
    s0_a = s0 # case when cat < k
    s0_b = s0_a.clone() # case when cat = k
    s0_b[0] = s0_b[1]

    # given k, and theta: find O and t_min
    # we want to match mean sig(O)to match S0 and var to be minimized 
    Oa = torch.tensor(O_init, requires_grad=True) # we know the true vector is onehot
    #f = torch.tensor(-O_init/np.sqrt(k), requires_grad=True)
    t_min = torch.tensor(t_init, requires_grad=True)

    # optimizer for O and t_min
    opt = torch.optim.Adam([Oa, t_min], lr=1e-2)

    # run until convergence
    loss_track = []
    for i in range(epochs):
        Oa_ = fhot(Oa, k)
        Ob_ = -Oa * torch.ones(k-1)
        S0_a = sample(Oa_, theta, t_min, k, N=N)
        S0_b = sample(Ob_, theta, t_min, k, N=N)

        # stack
        S0 = torch.stack((S0_a, S0_b), dim=0)

        # mean
        m = S0.mean(dim=1)
        std = S0.std(dim=1)

        # loss
        lma = (m[0] - s0_a).pow(2).mean()
        lmb = (m[1] - s0_b).pow(2).mean()
        lstd = (std - s0_b[1]).pow(2).mean()
        loss = lma + lmb + lstd

        # update O and t_min
        opt.zero_grad()
        loss.backward()
        opt.step()

        # if converged, exit
        loss_track.append(loss.item())
        if len(loss_track) > 100:
            loss_track.pop(0)
        if loss_track[-1] > loss_track[0]:
            print(f'case 1: {m[0].detach().numpy()}')
            print(f'case 2: {m[1].detach().numpy()}')
            print(f'std: {std.mean().detach()}')
            return Oa.detach(), t_min.detach()# , [m.detach(), v.detach(), sc.detach()]

        # no negative values allowed
        t_min.data = torch.clamp(t_min, min=1e-5)

# get t_max
def get_tmax(O, theta, k, t_init=1., N=1000, eps=1e-2, epochs=5000):
    t_max = torch.tensor(t_init, requires_grad=True)
    O = -O * torch.ones(k-1) # less biased

    # optimizer for h
    opt = torch.optim.Adam([t_max], lr=1e-2)

    # run until convergence
    loss_track = []
    for i in range(epochs):
        S = sample(O, theta, t_max, k, N=N)

        # make mean close to 1/k and t small
        mean = S.mean(dim=0)
        loss = (mean - 1/k).abs().mean()

        # update h
        opt.zero_grad()
        loss.backward()
        opt.step()

        # no negative values allowed
        t_max.data = torch.clamp(t_max, min=1e-5)

        # if converged, exit
        loss_track.append(loss.item())
        if len(loss_track) > 100:
            loss_track.pop(0)
        if loss_track[-1] > loss_track[0]:
            print(1-mean.sum().detach().item())
            print(mean.detach().numpy())
            return t_max.detach()

    # throw error if not converged
    raise ValueError(f'h did not converge after {epochs} iters')


if __name__ == '__main__':
    k=10; cat_mag=0.9

    # set initial value
    S0 = torch.zeros(k-1) + (1 - cat_mag)/(k-1)
    S0[0] = cat_mag

    # set final value
    S1 = torch.zeros(k-1) + 1/k

    # step 2: get h s.t.   
    theta = get_theta(k, sig3=0.1)
    print(f'theta: {theta:.5f}\n')

    # step 3: get Oa, Ob and t_min
    O, t_min = get_Ot(S0, theta, k)
    print(f'O: {O}')
    print(f't_min: {t_min.item():.5f}\n')

    # step 4: get t_max
    t_max = get_tmax(O, theta, k)
    print(f't_max: {t_max.item():.5f}')
