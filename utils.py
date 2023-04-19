import sys, os
import torch
import numpy as np
import shutil
import imageio
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage

def onehot2cat(x, k):
    return torch.argmax(x, dim=1) / (k-1)

# useful torch -> numpy
def ptnp(x):
    return x.detach().cpu().numpy()

# take t from [0, 1] to [t_min, t_max]
def scale_t(t, t_min, t_max):
    return t * (t_max - t_min) + t_min

# make gif from images, name is f'path/{}.png' from 0 to n
def make_gif(path, name, n):
        print('making gif...')
        images = []
        for i in range(n):
            images.append(imageio.imread(os.path.join(path, f'{i}.png')))
        imageio.mimsave(f'{name}.gif', images)

        # remove images and folder
        # shutil.rmtree('imgs')

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

def plot_score_norm(score_vec, t_vec, save_path="t_vs_score_norm.png"):
        # plot and save t vec vs score vec
        plt.plot(t_vec, [torch.linalg.vector_norm(score) for score in score_vec])
        plt.xlabel('t')
        plt.ylabel('score norm (fronorm)')
        plt.savefig(save_path)

def plot_score_std(score_vec, t_vec, save_path="t_vs_score_std.png"):
        plt.plot(t_vec, [torch.std(score) for score in score_vec])
        plt.xlabel('t')
        plt.ylabel('score std')
        plt.savefig(save_path)
