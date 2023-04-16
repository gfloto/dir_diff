import os, sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# check if function is monotonic
def is_monotonic(f, x):
    y = np.polyval(f, x)
    return np.all(np.diff(y) >= 0)

# sample time steps s.t. loss uniformly increases w time
class TimeSampler:
    def __init__(self, T, N=10, max_size=1000):
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
    def fit(self, order=4):
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

if __name__ == '__main__':
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

    ts.plot('test_sample.png')