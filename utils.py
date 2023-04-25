import sys, os
import argparse
import torch
import numpy as np
from einops import rearrange
from torch.nn.functional import one_hot
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    # add exp name
    parser.add_argument('--exp', type=str, default='dev', help='experiment name')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--proc_name', type=str, default='simplex', help='process name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--k', type=int, default=3, help='number of categories')
    parser.add_argument('--O', type=int, default=3, help='process origin')
    parser.add_argument('--h', type=int, default=10, help='process speed')
    parser.add_argument('--T', type=float, nargs='+', default=[0.08, 0.55], help='time interval')
    parser.add_argument('--a', type=int, default=10, help='simplex domain')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # asserts
    assert args.k > 1, 'k must be greater than 1'
    assert args.O > 0, 'O must be greater than 0'
    assert args.h > 0, 'h must be greater than 0'
    assert args.T[0] < args.T[1], 'T[0] must be less than T[1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'
    assert args.proc_name in ['cat', 'simplex'], 'process name must be cat or simplex'

    return args

def save_path(args, path):
    return os.path.join(args.exp, path)

# logit function with scale a
def a_logit(x, a):
    return x.log() - (a-x).log()

def onehot2cat(x, k):
    return torch.argmax(x, dim=1) / (k-1)

def cat2onehot(x, k):
    x = one_hot(x * (k-1), k).float() 
    return rearrange(x, 'b h w k -> b k h w')

# useful torch -> numpy
def ptnp(x):
    return x.detach().cpu().numpy()

class InfoLogger:
    def __init__(self):
        self.loss = None
        self.t = None

    def clear(self):
        self.loss = None
        self.t = None

    def store_loss(self, loss, t):
        t = t * np.ones_like(loss)

        if self.loss is None:
            self.loss = loss
            self.t = t
        else:
            self.loss = np.concatenate((self.loss, loss))
            self.t = np.concatenate((self.t, t))

    def plot_loss(self, path, sig=3, bins=50):
        t = self.t
        loss = self.loss

        # mean and std for edges
        t_mean = np.mean(t); t_std = np.std(t)
        l_mean = np.mean(loss); l_std = np.std(loss)

        # define edges
        xedges = np.linspace(np.min(t), np.max(t), bins+1)
        yedges = np.linspace(np.min(loss), np.max(loss), bins+1)

        # use numpy to compute 2d histogram
        H, xedges, yedges = np.histogram2d(loss, t, bins=(xedges, yedges))

        # plot 2d histogram
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, title='$log \mathcal{L}(t)$ Histogram',
                xlabel='t', ylabel='$log \mathcal{L}(t)$', 
                xlim=[np.min(t), np.max(t)], ylim=[np.min(loss), np.max(loss)])
        im = NonUniformImage(ax, interpolation='nearest')
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, H)
        ax.images.append(im)

        # save figure
        plt.savefig(path)
        plt.close()
