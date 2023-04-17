import sys, os
import torch
import argparse

from plot import save_vis
from model import Unet
from train import train
from train_utils import Process, TimeSampler
from dataloader import mnist_dataset

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--k', type=int, default=2, help='number of categories')
    parser.add_argument('--O', type=int, default=6, help='process origin')
    parser.add_argument('--h', type=int, default=4, help='process speed')
    parser.add_argument('--T', type=float, nargs='+', default=[0.1, 1], help='time interval')
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
    model = Unet(dim=64, channels=args.k).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    process = Process(args.O, args.h)
    time_sampler = TimeSampler()

    # train loop
    for epoch in range(args.epochs):
        train(model, process, loader, time_sampler, opt, args)
        print('done one loop')
        sys.exit()
