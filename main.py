import sys, os
import torch
from torch.nn.functional import one_hot
from einops import rearrange
import matplotlib.pyplot as plt

from plot import save_vis
from dataloader import mnist_dataset

if __name__ == '__main__':
    batch_size = 64
    k = 10 # number of categories (normally 255 for images)
    loader = mnist_dataset(batch_size, k)

    for i, (x, y) in enumerate(loader):
        print(x.shape, y.shape)
        break

        # show image
        #path = 'test.png'
        #x = save_vis(x, path, k)
        #break
    print('done')
