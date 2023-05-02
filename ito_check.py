import sys, os
from tqdm import tqdm
import torch
import numpy as np
from einops import repeat, rearrange
import matplotlib.pyplot as plt

from plot import make_gif

def sig(y):
    x_ = y.exp() / (1 + y.exp().sum(dim=1, keepdim=True))
    return x_

# generalized inverse sigmoid
def sig_inv(x_):
    xd = 1 - x_.sum(dim=1, keepdim=True)
    y = x_.log() - xd.log()
    return y

# plot 2d or 3d scatter plot of distributions
def plot(x, i, path):
    d = x[0].shape[1]
    if d == 2:
        # plot 2d
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.scatter(x[0][:,0], x[0][:,1], s=2)
        plt.scatter(x[1][:,0], x[1][:,1], s=2)

    if d == 3:
        # plot 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0.1*i, i-90)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        ax.scatter(x[0][:, 0], x[0][:, 1], x[0][:,2], s=2)
        ax.scatter(x[1][:, 0], x[1][:, 1], x[1][:,2], s=2)

    #if i % 10 == 0: plt.show()
    plt.legend(['OU', 'S'])
    plt.savefig(path)
    plt.close()

# OU process in gaussian space, mapped back to simplex
class GProcess:
    def __init__(self, m0, theta, N):
        self.N = N
        self.theta = theta
        self.m0 = m0
        self.d = m0.shape[0]

    def __call__(self, t):
        mu = self.m0 * (-self.theta * t).exp() 
        var = 1/(2*self.theta) * (1 - (-2*self.theta * t).exp())

        sample = var.sqrt() * torch.randn(self.N, self.d) + mu
        return sig(sample)

# corresponding process on simplex
class SProcess:
    def __init__(self, s0, theta, steps=10):
        self.steps = steps
        self.theta = theta
        self.d = s0.shape[0]
        self.s = repeat(s0, 'd -> N d', N=N) 
    
    # -theta * grad^T sig^-1(S)dt + 0.5h
    # this steps s by dt! not the same as GProcess
    def __call__(self, dt):
        f = self.sde_f(self.s)
        g = self.sde_g(self.s)

        # g from ds = f dt + g dB 
        dB = np.sqrt(dt) * torch.randn_like(self.s)

        # update s
        gdB = torch.einsum('b i j, b j -> b i', g, dB)

        update = f*dt + gdB
        self.s = self.s + update

        return self.s

    # make drift term sde
    def sde_f(self, s):
        b, k = s.shape[:2]

        x = sig_inv(self.s)
        beta = -self.theta*x + 0.5*(1-2*s)

        # \sum_{i \neq j} X_{ij}, vectorized
        beta_v = repeat(s*beta, 'b d -> b k d', k=self.d)
        I = repeat(torch.eye(k), 'i j -> b i j', b=b).to(s.device)
        bsum = torch.einsum('b i j, b i j -> b i', 1-I, beta_v)

        f = s * ( (1-s)*beta - bsum )
        return f

    # make diffusion term sde
    def sde_g(self, s):
        b, k = s.shape[:2]
        I = repeat(torch.eye(k), 'i j -> b i j', b=b).to(s.device)

        neq = torch.einsum('b i, b j -> b i j', s, s)
        eq = repeat(s*(1-s), 'b d -> b k d', k=self.d)

        g1 = torch.einsum('b i j, b i j -> b i j', I, eq)
        g2 = torch.einsum('b i j, b i j -> b i j', 1-I, neq)

        g = g1 - g2
        return g


if __name__ == '__main__':
    t1 = 1
    T = 50; N = 1000; k = 3
    theta = torch.rand(1)
    m0 = torch.randn(k-1)
    s0 = sig(m0[None, ...])[0]

    sproc = SProcess(s0, theta)
    gproc = GProcess(m0, theta, N)

    os.makedirs('imgs', exist_ok=True)
    t = np.linspace(0, t1, T)
    for i in tqdm(range(T)):
        sxt = sproc(t1/T)
        gxt = gproc(t[i])

        # plot together to compare
        plot([gxt, sxt], i, f'imgs/{i}.png')

    make_gif('imgs', 'ito_check.gif', T)

