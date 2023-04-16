import sys, os
import torch
from torch.nn.functional import one_hot
from einops import rearrange
import matplotlib.pyplot as plt

from plot import save_vis
from model import Unet
from train import TimeSampler, Process
from dataloader import mnist_dataset

def train(model, loader, optimizer, device, k):
    model.train()
    for i, (x, y) in enumerate(loader):
        x_0 = x.to(device)
        t = torch.rand(1).to(device)

        # sample x_t
        x_t = 

        # forward pass
        x_out = model(x, t)

        # show image
        path = 'test.png'
        x = save_vis(x, path, k)
        break

if __name__ == '__main__':
    lr = 1e-3
    batch_size = 64
    k = 2 # number of categories (normally 255 for images)
    O = 6 # process origin
    h = 4 # process speed?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    # load dataset
    loader = mnist_dataset(batch_size, k)
    model = Unet(dim=64, channels=k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    process = Process(O, h)

    # train loop
    while True:
        train(model, loader, opt, device, k)
