import os, sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import onehot2cat, a_logit
from plot import save_vis
from dataloader import mnist_dataset

'''
main diffusion process class
'''

class Process:
    def __init__(self, args):
        self.O = torch.tensor(args.O).to(args.device)
        self.h = torch.tensor(args.h).to(args.device)
        self.a = torch.tensor(args.a).to(args.device)
        self.t_min, self.t_max = args.T
        self.device = args.device

    # get t, rescale to be in proper interval
    def t(self):
        tu = torch.rand(1)
        t = (self.t_max - self.t_min) * tu + self.t_min
        return t.to(self.device), tu.to(self.device)

    # mean and variance of OU process at time t
    def mean_var(self, x0, t):
        mu = self.O * torch.exp(-self.h*t) * x0
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*t)) * torch.ones_like(x0).to(x0.device)
        return mu, var

    # sample from logit normal distribution
    def sample(self, mu, var):
        return self.a * torch.sigmoid(mu + var.sqrt()*torch.randn_like(mu))

    # return xt given x0 and t
    def xt(self, x0, t):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance, then sample
        mu, var = self.mean_var(x0, t)
        xt = self.sample(mu, var)
        return xt

    # compute score at xt given mu and var
    def s(self, xt, mu, var):
        num = a_logit(xt, self.a) - 2*var*xt - mu + self.a*var
        denom = var*xt*(xt - self.a)
        score = num / denom
        return score

    # score at xt given x0 and t
    def score(self, x0, xt, t):
        # convert x0 to either -1 or 1
        x0 = onehot2cat(x0, k=2) * 2 - 1

        # get mean and variance
        mu, var = self.mean_var(x0, t)
        score = self.s(xt, mu, var)

        return score


'''
testing scripts for process and time sampler
'''

from tqdm import tqdm
from plot import make_gif
plt.style.use('seaborn-whitegrid')

if __name__ == '__main__':
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
    imgs = []; r_track = []
    for i in tqdm(range(T[2])):
        xt = process.xt(x0, t[i])
        score, r = process.score(x0, xt, t[i], norm=True)
        r_track.append(r.cpu().detach().numpy())

        # save image
        #save_vis(xt, f'imgs/{i}.png', k=None, n=n)

    # plot r track
    plt.plot(np.linspace(*T), r_track)
    plt.xlabel('Time'); plt.ylabel('Score Magnitude')
    plt.title('Score Magnitude vs Time')
    plt.show()

    # make gif of forward process
    make_gif('imgs', 'results/forward.gif', T[2])


