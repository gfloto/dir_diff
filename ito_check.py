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

# batched vector -> diagonal matrix with else c 
def diag_d(X, c): # matrix x[b,k,h,w], scalar c
    b, k = X.shape

    # pattern to keep diagonal entries, else 1
    E_ = torch.eye(k, device=X.device)
    E = repeat(E_, 'i j -> b i j', b=b)
    Y = torch.einsum('b i, b i j -> b i j', X-c, E)
    Y = Y + c # do this to keep c on non-diagonal entries

    return Y

# batched vector -> repated matrix with c on diagonals 
def diag_u(X, c): # matrix x[b,k,h,w], scalar c
    b, k = X.shape
    X_ = repeat(X-c, 'b k -> b k d', d=k)

    # pattern to make diagonal entries c, else same
    E_ = 1 - torch.eye(k, device=X.device)
    E = repeat(E_, 'i j -> b i j', b=b)
    Y = torch.einsum('b i, b i j -> b i j', X-c, E)
    Y = Y + c # do this to keep c on non-diagonal entries

    return Y

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

class OProcess:
    def __init__(self, x0, theta):
        self.theta = theta
        self.d = x0.shape[0]
        self.x = repeat(x0, 'd -> N d', N=N) 

    def __call__(self, dt):
        B = np.sqrt(dt) * torch.randn_like(self.x)
        update = -self.theta*self.x*dt + B
        self.x = self.x + update

        return self.x

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
        jac = self.jac(self.s)
        hess = self.hess(self.s)

        # x and noise to be added
        x = sig_inv(self.s)

        # f from ds = f dt + g dB 
        fx = torch.einsum('b i j, b j-> b i', jac, x)
        f = -self.theta*fx + 0.5*hess

        # g from ds = f dt + g dB 
        B = np.sqrt(dt) * torch.randn_like(self.s)
        diff = torch.einsum('b i j, b j -> b i', jac, B) 

        # update s
        update = f*dt + diff

        self.s = self.s + update
        return self.s

    # make jac term for sde
    def jac(self, s):
        f1 = repeat(s, 'b k -> b k d', d=self.d) 
        f2 = diag_d(1-s, 1)
        f3 = rearrange(diag_u(-s, 1), 'b i j -> b j i')

        J = f1 * f2 * f3
        return J

    # make hess term for sde
    def hess(self, s):
        f1 = s * (1-s) * (1 - 2*s)

        m = diag_u(s, 0)
        b_ = s * (1 - 2*s)
        b = repeat(b_, 'b k -> b k d', d=self.d)
        b = rearrange(b, 'b i j -> b j i')
        f2 = torch.einsum('b i j, b i j -> b j', m, b)

        #print(s)
        #print(m)
        #print(b)
        #print(f2)
        #sys.exit()

        h = f1 - f2
        return h


if __name__ == '__main__':
    t1 = 0.5
    T = 50; N = 1000; k = 4
    theta = torch.rand(1)
    m0 = torch.randn(k-1)
    #m0 = torch.tensor([1, 0.5, 0])
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

