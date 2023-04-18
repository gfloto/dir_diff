import os, sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import onehot2cat, scale_t
from plot import save_vis
from dataloader import mnist_dataset

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

    def s(self, xt, mu, var):
        num = torch.logit(xt) - 2*var*xt - mu + var
        denom = var*xt*(xt-1)
        score = num / denom
        return score

    # score at xt given x0 and t
    def score(self, x0, xt, t, norm=True):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance
        mu, var = self.mean_var(x0, t)
        score = self.s(xt, mu, var)

        r = self.score_scale(mu, var, t)
        return score / r

    # normalize by average score at 1 std. dev.
    def score_scale(self, mu, var, t):
        # only need to compute for one value of t
        mu = mu[0,0,0]; var = var[0,0,0];

        # get values at +- sigma
        b1 = torch.sigmoid(mu - torch.sqrt(var))
        b2 = torch.sigmoid(mu + torch.sqrt(var))

        # get score at +- sigma
        s1 = self.s(b1, mu, var)
        s2 = self.s(b2, mu, var)

        r = (s1.abs() + s2.abs()) / 2
        return r

'''
time sampler from: https://arxiv.org/abs/2211.15089 
'''

# check if function is monotonic
def is_monotonic(f, x):
    y = np.polyval(f, x)
    return np.all(np.diff(y) >= 0)

# sample time steps s.t. loss uniformly increases w time
class TimeSampler:
    def __init__(self, t_min, t_max, device, N=10, max_size=4096):
        self.t_min = torch.tensor(t_min)
        self.t_max = torch.tensor(t_max) # time step range
        self.N = N # number of bins-1 (f is piecewise linear)
        self.max_size = max_size # max size of data to store for fitting f
        self.device = device

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
            u = torch.rand(1)
            t = scale_t(u, self.t_min, self.t_max).to(self.device)
            return t 

        # sample u, find index of bin u is in
        u = (self.f_(-1) - self.f_(0)) * np.random.rand() + self.f_(0)
        loc = np.searchsorted(self.f_(), u)

        # to inverse cdf to get t
        # l = f(e0) + m*(t - e0), m = (f(e1) - f(e0)) / (e1 - e0)
        # t = (l - f(e0)) / m + e0
        m = self.f_(loc) - self.f_(loc-1)
        m /= self.edges[loc] - self.edges[loc-1]
        t = (u - self.f_(loc-1)) / m + self.edges[loc-1]
        
        # rescale to proper range
        t = scale_t(t, self.t_min, self.t_max) 
        return torch.tensor(t).to(self.device)

    # fit f to data 
    def fit(self, order=2):
        if self.data is None:
            self.f = None; self.edges = None
            return
        else: # sorta data by t to check monotonicity
            data = self.data[self.data[:,0].argsort()]

        # fit polynomial to data f(t) = loss
        loss, t = data[:,0], data[:,1]
        self.f = np.polyfit(t, loss, order)

        # ensure function is monotonic
        if not is_monotonic(self.f, t):
            print('function is not monotonic')
            # plot data and function
            x = np.linspace(self.t_min, self.t_max, 1000)
            y = np.polyval(self.f, x)
            plt.plot(x, y)
            plt.scatter(t, loss)
            plt.savefig('error.png')
            plt.close()
            self.f = None; self.edges = None;
            return

        # n-1 evenly spaced bins, spaced s.t. f(t)_i - f(t)_{i-1} is constant 
        x = np.linspace(self.t_min, self.t_max, 100*self.N)
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
    def update(self, loss, t):
        # add new data
        t = t * np.ones_like(loss)
        if self.data is None:
            self.data = np.array([loss, t]).T
        else:
            self.data = np.append(self.data, np.array([loss, t]).T, axis=0)

        # ensure data is of size max_size
        if self.data.shape[0] > self.max_size:
            a = self.data.shape[0] - self.max_size
            self.data = self.data[a:]

    # plot data and f
    def plot(self, path):
        if self.data is None or self.f is None: return

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

from utils import make_gif

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
        imgs = []
        for i in range(T[2]):
            xt = process.xt(x0, t[i])

            # save image
            save_vis(xt, f'imgs/{i}.png', k=None, n=n)

        # make gif of forward process
        make_gif('imgs', 'results/forward.gif', T[2])

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

        ts.plot('results/ts_sample.png')