import os, sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

plt.style.use('seaborn')

def onehot2cat(x, k):
    return torch.argmax(x, dim=1) / (k-1)

# visualize images
def save_vis(x, path, k, x_out=None, n=8):
    x = onehot2cat(x, k=k)
    if x_out is not None:
        x_out = onehot2cat(x_out, k=10)
    else:
        x_out = x
    print(torch.unique(x_out))

    # take first n images
    n = min(n, x.shape[0])
    x = x[:n]; x_out = x_out[:n]

    # x in top row, x_out in bottom row
    # stitch using einops
    x = rearrange(x, 'b h w -> h (b w)', b=n)
    x_out = rearrange(x_out, 'b h w -> h (b w)', b=n)
    img = torch.stack((x, x_out))
    img = rearrange(img, 'b h w -> (b h) w', b=2)

    # convert to numpy and save
    img = img.detach().cpu().numpy()

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