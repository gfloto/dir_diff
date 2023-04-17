import os, sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import onehot2cat
from plot import save_vis
from dataloader import mnist_dataset

# score of logit normal distribution (d/dx log pdf)
def score(x, mu, var):
    num = logit(x) - 2*var*x - mu + var
    denom = var*x*(x-1)
    return num / denom

# pdf of logit normal distribution
def pdf(x, mu, v):
    return 1/np.sqrt(2*np.pi*v) * np.exp(-0.5/v * (logit(x) - mu)**2) / (x * (1-x))

'''
main diffusion process class
'''

class Process:
    def __init__(self, O, h, device):
        self.O = torch.tensor(O).to(device)
        self.h = torch.tensor(h).to(device)

    # mean and variance of process at time t
    # NOTE: currently x0 is either -1 or 1
    def mean_var(self, x0, t):
        mu = self.O * torch.exp(-self.h*t) * x0
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*t)) * torch.ones_like(x0).to(x0.device)
        return mu, var

    # sample from logit normal distribution
    def sample(self, mu, var):
        return torch.sigmoid(var.sqrt() * torch.randn_like(mu) + mu)

    # return xt given x0 and t
    # TODO: this will be different when d > 2
    def xt(self, x0, t):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance, then sample
        mu, var = self.mean_var(x0, t)
        xt = self.sample(mu, var)

        return xt


'''
time sampler from: https://arxiv.org/abs/2211.15089 
'''

# check if function is monotonic
def is_monotonic(f, x):
    y = np.polyval(f, x)
    return np.all(np.diff(y) >= 0)

# sample time steps s.t. loss uniformly increases w time
class TimeSampler:
    def __init__(self, T=1, N=10, max_size=1000):
        self.T = T # length of process
        self.N = N # number of bins-1 (f is piecewise linear)
        self.max_size = max_size # max size of data to store for fitting f

        self.data = None # np array of (t, loss) tuples
        self.edges = None # bin edges for inverse cdf
        self.f = None # function from polyfit
    
    # call np.polyfit on data
    def f_(self, i=None):
        if i is not None:
            return np.polyval(self.f, self.edges[i])
        else:
            return np.polyval(self.f, self.edges)

    # get sample of t s.t. loss uniformly increases w.r.t. t
    def __call__(self):
        # t = f^-1(u) where u ~ U(0, L)
        if self.f is None or self.edges is None:
            return np.random.rand()

        # sample u, find index of bin u is in
        u = (self.f_(-1) - self.f_(0)) * np.random.rand() + self.f_(0)
        loc = np.searchsorted(self.f_(), u)

        # to inverse cdf to get t
        # l = f(e0) + m*(t - e0), m = (f(e1) - f(e0)) / (e1 - e0)
        # t = (l - f(e0)) / m + e0
        m = self.f_(loc) - self.f_(loc-1)
        m /= self.edges[loc] - self.edges[loc-1]
        t = (u - self.f_(loc-1)) / m + self.edges[loc-1]

        return t

    # fit f to data 
    def fit(self, order=6):
        if self.data is None:
            self.f is None or self.edges is None;
            return
        else: # sorta data by t to check monotonicity
            self.data = self.data[self.data[:,0].argsort()]

        # fit polynomial to data f(t) = loss
        t, loss = self.data[:,0], self.data[:,1]
        self.f = np.polyfit(t, loss, order)

        # ensure function is monotonic
        if not is_monotonic(self.f, t):
            self.f is None or self.edges is None;
            print('function is not monotonic')
            return

        # n-1 evenly spaced bins, spaced s.t. f(t)_i - f(t)_{i-1} is constant 
        x = np.linspace(0, self.T, 100*self.N)
        y = np.polyval(self.f, x)

        # get bin edges
        last_val = 0
        rise = (y[-1] - y[0]) / self.N
        self.edges = np.array([x[0]])
        for i in range(y.shape[0]):
            if y[i] - last_val >= rise:
                last_val = y[i]
                self.edges = np.append(self.edges, [x[i]])
        self.edges = np.append(self.edges, [x[-1]])

    # add new data to sampler 
    def update(self, t, loss):
        # add new data
        if self.data is None:
            self.data = np.array([(t, loss)])
        else:
            self.data = np.append(self.data, [(t, loss)], axis=0)

        # ensure data is of size max_size
        if len(self.data) > self.max_size:
            self.data = self.data[1:]

    # plot data and f
    def plot(self, path):
        if self.data is None: return

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.plot(self.data[:,0], self.data[:,1], 'o')
        ax.plot(self.edges, self.f_(), 'r')

        x = np.linspace(0, self.T, 1000)
        ax.plot(x, np.polyval(self.f, x), 'b')

        # plot bin edges
        for i in range(len(self.edges)):
            plt.axvline(self.edges[i], color='k', linestyle='--')

        plt.savefig(path)

# args are either proc or ts
def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--test', type=str, default='proc')
    args = args.parse_args()

    assert args.test in ['proc', 'ts']
    return args 

'''
testing scripts for process and time sampler
'''

import imageio
import shutil

if __name__ == '__main__':
    args = get_args()

    # save video of forward process
    if args.test == 'proc':
        # get device, data and process
        h = 8; O = 6
        n = 8; k=2; T = [0.075, 0.7, 100]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader = mnist_dataset(n, k)
        process = Process(O, h, device)

        # test forward process
        t = torch.linspace(*T).to(device)
        (x0, _) = next(iter(loader))
        x0 = x0.to(device)

        # get forward process at each t
        print('running forward process...')
        os.makedirs('imgs', exist_ok=True)
        for i in range(T[2]):
            xt = process.xt(x0, t[i])

            # save image
            save_vis(xt, f'imgs/{i}.png', k=None, n=n)

        # make gif of forward process
        print('making gif...')
        images = []
        for i in range(T[2]):
            images.append(imageio.imread(f'imgs/{i}.png'))
        imageio.mimsave('forward.gif', images)

        # remove images and folder
        #shutil.rmtree('imgs')

    elif args.test == 'ts':
        # test time sampler
        print('testing time sampler...')
        ts = TimeSampler(1)

        for i in range(1000):
            t = np.random.rand()
            loss = np.tanh(3*t) + 1 
            ts.update(t, loss)

        import time
        t0 = time.time()
        ts.fit()
        out = ts()
        print(f'fit time: {time.time() - t0:.5f} s')

        ts.plot('ts_sample.png')