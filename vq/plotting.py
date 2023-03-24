import os, sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

plt.style.use('seaborn')

# visualize images
def save_vis(x, x_out, epoch, path, n=8):
    # take first n images
    x = x[:n]; x_out = x_out[:n]

    # x in top row, x_out in bottom row
    # stitch using einops
    x = rearrange(x, 'b c h w -> h (b w) c', b=n)
    x_out = rearrange(x_out, 'b c h w -> h (b w) c', b=n)
    img = torch.stack((x, x_out))
    img = rearrange(img, 'b h w c -> (b h) w c', b=2)

    # convert to numpy and save
    img = img.detach().cpu().numpy()
    img = Image.fromarray(np.uint8(img*255))
    img.save(os.path.join(path, f'sample_{epoch}.png'))

# plot loss, use log scale 
def plot_loss(loss, path):
    plt.plot(loss)
    plt.yscale('log')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.clf()