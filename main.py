import sys, os
import torch
import argparse

import matplotlib.pyplot as plt

from plot import save_vis
from model import Unet
from train import train
from train_utils import Process, TimeSampler
from dataloader import mnist_dataset
from utils import InfoLogger

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--k', type=int, default=2, help='number of categories')
    parser.add_argument('--O', type=int, default=6, help='process origin')
    parser.add_argument('--h', type=int, default=8, help='process speed')
    parser.add_argument('--T', type=float, nargs='+', default=[0.075, 0.7], help='time interval')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # asserts
    assert args.k > 1, 'k must be greater than 1'
    assert args.O > 0, 'O must be greater than 0'
    assert args.h > 0, 'h must be greater than 0'
    assert args.T[0] < args.T[1], 'T[0] must be less than T[1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'

    return args


if __name__ == '__main__':
    args = get_args()
    print(f'device: {args.device}')

    # load dataset, model, optimizer and process
    loader = mnist_dataset(args.batch_size, args.k)
    # TODO: for now k-1
    model = Unet(dim=64, channels=args.k-1).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    process = Process(args.O, args.h, args.device)
    time_sampler = TimeSampler(*args.T, args.device)
    logger = InfoLogger()

    # train loop
    loss_track = []
    for epoch in range(args.epochs):
        logger.clear()
        loss = train(model, process, loader, time_sampler, opt, logger, args)

        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # plot loss
        plt.plot(loss_track)
        plt.savefig('loss.png')
        plt.close()
