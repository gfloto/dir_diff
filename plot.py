import os, sys
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt

from utils import onehot2cat

# visualize images
def save_vis(x, path, k, x_out=None, n=8):
    # convert from onehot to categorical
    if len(x.shape) == 4:
        x = onehot2cat(x, k=k)
        x = x / (k-1)
    if x_out is None:
        x_out = torch.zeros_like(x)
    elif len(x_out.shape) == 4:
        x_out = onehot2cat(x_out, k=10)
        x_out = x_out / (k-1)

    # take first n images
    n = min(n, x.shape[0])

    # x in top row, x_out in bottom row
    # stitch using einops
    x = rearrange(x[:n], 'b h w -> h (b w)', b=n)
    if x_out is not None:
        x_out = rearrange(x_out[:n], 'b h w -> h (b w)', b=n)
        x = torch.stack((x, x_out))
        x = rearrange(x, 'b h w -> (b h) w', b=2)

    # convert to numpy and save
    img = x.detach().cpu().numpy()

    # save image
    fig = plt.figure(figsize=(2*n, 4))
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