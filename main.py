import sys, os
import torch
from torch.nn.functional import one_hot
from einops import rearrange
import matplotlib.pyplot as plt

from plot import save_vis
from model import Unet
from dataloader import mnist_dataset

if __name__ == '__main__':
    batch_size = 64
    k = 2 # number of categories (normally 255 for images)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load dataset
    loader = mnist_dataset(batch_size, k)

    # load model
    model = Unet(dim=64, channels=k).to(device)

    # train loop
    for i, (x, y) in enumerate(loader):
        t = torch.rand(1).to(device)
        x = x.to(device)

        # forward pass
        x_out = model(x, t)
        print(f'x_out: {x_out.shape} x: {x.shape}')

        # show image
        path = 'test.png'
        x = save_vis(x, path, k)
        break
    print('done')
