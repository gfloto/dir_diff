import os, sys
import shutil
import torch
import numpy as np
import imageio
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt

from utils import onehot2cat

# make gif from images, name is f'path/{}.png' from 0 to n
def make_gif(path, name, n):
        print('making gif...')
        images = []
        for i in range(n):
            images.append(imageio.imread(os.path.join(path, f'{i}.png')))
        imageio.mimsave(f'{name}', images)

        # remove images and folder
        shutil.rmtree('imgs')

# visualize images
# x can be list of tensors or tensor
def save_vis(x, path, a=1, k=None, n=8):
    # if not list, make list (to make process the same)
    if not isinstance(x, list):
        x = [x]
    else: # check dim 0 > n (to make collage)
        for i in range(len(x)):
            assert x[i].shape[0] >= n

    # ensure x is in [0, 1]
    for i in range(len(x)):
        x[i] = x[i] / a
        assert torch.all(x[i] >= 0) and torch.all(x[i] <= 1)

        # convert from onehot to categorical if required
        if len(x[i].shape) == 4: # ie. [b, k, h, w]
            assert k is not None
            x = onehot2cat(x, k=k)
            x = x / (k-1)

    # stitch list using einops
    imgs = len(x)
    for i in range(len(x)):
        x[i] = rearrange(x[i][:n], 'b h w -> h (b w)', b=n)
    x = torch.stack(x)
    x = rearrange(x, 'b h w -> (b h) w', b=imgs)

    # convert to numpy and save
    img = x.detach().cpu().numpy()

    # save image
    fig = plt.figure(figsize=(2*n, 4*imgs))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.savefig(path)
    plt.close()

# plot loss, use log scale 
def plot_loss(loss, path):
    plt.plot(loss)
    plt.yscale('log')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.clf()