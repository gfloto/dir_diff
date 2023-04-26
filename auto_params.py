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
def sig_inv(x_, a):
    xd = a - x_.sum(dim=1, keepdim=True)
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

# TODO: include derivation that goes with this
# step 1: get simplex size given d, pad g and dist q (rename)
def get_af(d, pad=3, dist=9):
    # 1. cushion 'pad' on all sides on simplex
    # 2. length 'dist' from state at t_min to t_max
    dist = dist**2
    r = np.sqrt(dist * (d-1)) / np.sqrt(d)
    a1 = (-d*pad - r + pad) / (1/d - 1)
    a2 = (-d*pad + r + pad) / (1/d - 1)

    # take smallest possible value 
    m = min(a1, a2)
    if (m > 0 and m - d*pad > 0):
        a = m
    else:
        a = max(a1, a2)
    
    # set value and return
    f = a - d*pad
    return a, f

# TODO: generalize this whole tings into a optimization class...

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

# one hot in first position
def fhot(O):
    # one hot
    onht = torch.zeros(d-1)
    onht[0] = 1
    return O * onht

# helpful function to sample from OU process
def sample(O, h, t, a, d, N=1000):
    # get mean and variance of OU process
    mu = O*(-h*t).exp()
    var = 1/(2*h) * (1 - (-2*h*t).exp())

    # sample in R, then map to simplex
    sample = var.sqrt() * torch.randn(N, d-1) + mu
    X0 = sig(sample, a)
    return X0, mu, var

# step 3: get O and t_min
def get_Ot(x0, a, h, O_init=3., t_init=0.1, N=1000, epochs=5000):
    d = x0.shape[0]
    x0_a = x0[:-1] # case when k < d
    x0_b = x0_a.clone() # case when k = d
    x0_b[0] = x0_b[1]

    # given a, d, and h: find O and t_min
    # we want to match mean to match x0 and var to be minimized 
    Oa = torch.tensor(O_init, requires_grad=True) # we know the true vector is onehot
    f = torch.tensor(-O_init/np.sqrt(d), requires_grad=True)
    t_min = torch.tensor(t_init, requires_grad=True)

    # optimizer for O and t_min
    opt = torch.optim.Adam([Oa, f, t_min], lr=1e-2)

    # run until convergence
    loss_track = []
    for i in range(epochs):
        Oa_ = fhot(Oa)
        Ob_ = f* Oa * torch.ones(d-1)
        X0_a, mu_a, var_a = sample(Oa_, h, t_min, a, d, N=N)
        X0_b, mu_b, var_b = sample(Ob_, h, t_min, a, d, N=N)

        # stack on new dimension
        X0 = torch.stack([X0_a, X0_b], dim=0)

        # get mean and std
        m = X0.mean(dim=1)
        v = X0.var(dim=1)

        # loss
        l1a = (m[0] - x0_a).pow(2).mean()
        l1b = (m[1] - x0_b).pow(2).mean()
        l2 = (v - 1).pow(2).mean()
        loss = l1a + l1b + l2

        # update O and t_min
        opt.zero_grad()
        loss.backward()
        opt.step()

        # if converged, exit
        loss_track.append(loss.item())
        if len(loss_track) > 100:
            loss_track.pop(0)
        if loss_track[-1] > loss_track[0]:
            #sc = score(X0, mu=mu, v=var, a=a).abs().mean()
            return Oa.detach(), f.detach(), t_min.detach()# , [m.detach(), v.detach(), sc.detach()]

        # no negative values allowed
        t_min.data = torch.clamp(t_min, min=1e-5)

# get t_max
def get_tmax(Oa, a, h, d, t_init=1., N=1000, eps=1e-2, epochs=5000):
    t_max = torch.tensor(t_init, requires_grad=True)

    # optimizer for h
    opt = torch.optim.Adam([t_max], lr=1e-2)

    # run until convergence
    loss_track = []
    for i in range(epochs):
        X0, _, _ = sample(Oa, h, t_max, a, d, N=N)

        # make mean close to a/2 and t small
        mean = X0.mean(dim=0)
        l1 = (mean - a/d).pow(2).mean()
        l2 = t_max.pow(2)
        loss = l1 + l2

        # update h
        opt.zero_grad()
        loss.backward()
        opt.step()

        # no negative values allowed
        t_max.data = torch.clamp(t_max, min=1e-6)

        # if converged, exit
        loss_track.append(loss.item())
        if len(loss_track) > 100:
            loss_track.pop(0)
        if loss_track[-1] > loss_track[0]:
            return t_max.detach(), mean

    # throw error if not converged
    raise ValueError(f'h did not converge after {epochs} iters')



if __name__ == '__main__':
    d = 4; pad = 3; dist = 6 / np.sqrt(d)

    # step 1: get a :and x0
    a, f = get_af(d, pad, dist)
    print('step: 1')
    print(f'a: {a:.5f}, x0: [{f + pad:.5f}, {pad}, ...]')

    # set initial value
    x0 = torch.zeros(d) + pad
    x0[0] += f

    # set final value
    x1 = torch.zeros(d) + a/d

    # check that sum is close
    print(f'||x0||: {x0.sum().item():.5f} - should be {a:.5f}')
    print(f'd(x0, x1): {(x0 - x1).pow(2).sum().sqrt().item():.5f} - should be {dist:.5f}\n')

    # step 2: get h
    h = get_h(d, a)
    print('step: 2')
    print(f'h: {h:.5f}')

    # check average score at t -> inf
    var = 1/(2*h)
    noise = torch.randn(10000, d-1)
    y = noise * var.sqrt()
    x = sig(y, a)
    sc = score(x, mu=0, v=1/(2*h), a=a)

    print(f'avg score: {sc.abs().mean().item():.5f}\n')

    # step 3: get O and t_min
    Oa, f, t_min = get_Ot(x0, a, h)
    print('step: 3')
    print(f'Oa: {Oa.item():.5f}, t_min: {t_min.item():.5f}')
    print(f'Ob: {f * Oa}\n')

    # step 4: get t_max
    t_max, mean = get_tmax(Oa, a, h, d)
    print('step: 4')
    print(f't_max: {t_max.item():.5f}')
    print(f'mean error: {mean.abs().mean() - a/d:.5f}')

