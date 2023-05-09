import os, sys
import shutil
import torch
import numpy as np
import imageio
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
plt.style.use('seaborn')

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
def save_vis(x, path, k, n=8):
    # if not list, make list (to make process the same)
    if not isinstance(x, list):
        x = [x]
    else: # check dim 0 > n (to make collage)
        for i in range(len(x)):
            assert x[i].shape[0] >= n

    # ensure x is in [0, 1]
    for i in range(len(x)):
        assert torch.all(x[i] >= 0)
        assert torch.all(x[i] <= 1)

        # convert from onehot to categorical
        xd = 1 - torch.sum(x[i], dim=1, keepdim=True)
        x[i] = torch.cat((x[i], xd), dim=1)

        # convert to categorical
        x[i] = onehot2cat(x[i], k=k)
        x[i] = x[i] / (k-1)

    # stitch list using einops
    imgs = len(x)
    for i in range(len(x)):
        if len(x[i].shape) == 3: # ie. [b, h, w]
            x[i] = rearrange(x[i][:n], 'b h w -> h (b w)', b=n)
        elif len(x[i].shape) == 4: # ie. [b, k, h, w]
            x[i] = rearrange(x[i][:n], 'b c h w -> c h (b w)', b=n)

    # reshape into final image
    x = torch.stack(x)
    if len(x.shape) == 3:
        x = rearrange(x, 'b h w -> (b h) w', b=imgs)
    elif len(x.shape) == 4:
        x = rearrange(x, 'b c h w -> (b h) w c', b=imgs)

    # convert to numpy and save
    img = x.detach().cpu().numpy()

    # save image
    fig = plt.figure(figsize=(2*n, 2*imgs))
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# plot loss, use log scale 
def plot_loss(loss, path):
    plt.plot(loss)
    plt.yscale('log')
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()

# hist plot for tracking loss vs time
def hist_plot(x, y, path):
    y /= np.mean(y)
    plt.scatter(x, y)

    # remove outliers above 3 std
    y = np.exp(y)
    x = np.array(x)
    inds = np.where(y < np.mean(y) + 3*np.std(y))[0]
    y = y[inds]
    x = x[inds]

    plt.scatter(x, y / np.mean(y))
    plt.xlabel('time')
    plt.ylabel('loss')
    plt.savefig(path)
    plt.close()

# plot 2d or 3d scatter plot of distributions
def scatter_plot(x, i, path):
    # make list if not list
    if not isinstance(x, list):
        x = [x]

    d = x[0].shape[1]
    if d == 2:
        # plot 2d
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for j in range(len(x)):
            ax.scatter(x[j][:,0], x[j][:,1], s=2)

    if d == 3:
        # plot 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0.1*i+10, i-45)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        for j in range(len(x)):
            ax.scatter(x[j][:,0], x[j][:,1], x[j][:,2], s=2)

    #if i % 10 == 0: plt.show()
    plt.savefig(path)
    plt.close()

# plot 2d or 3d vector field of distributions
def vector_plot(x, i, path):
    # make list if not list
    if not isinstance(x, list):
        x = [x]
    
    d = x[0].shape[2]
    if d == 2:
        # plot 2d
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # add color from default color cycle
        for j in range(len(x)):
            ax.quiver(x[j][0,:,0], x[j][0,:,1], x[j][1,:,0], x[j][1,:,1], 
                       scale=1, scale_units='xy', angles='xy', width=0.002, color=f'C{j}')

    if d == 3:
        # plot 3d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(0.1*i+10, i-45)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # add color from default color cycle
        for j in range(len(x)):
            ax.quiver(x[j][0,:,0], x[j][0,:,1], x[j][0,:,2], x[j][1,:,0], x[j][1,:,1], x[j][1,:,2],
                       color=f'C{j}')

    #if i % 10 == 0: plt.show()
    plt.savefig(path)
    plt.close()