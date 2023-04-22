import sys, os
import torch
import json
import argparse

import matplotlib.pyplot as plt

from plot import save_vis
from model import Unet
from train import train, cat_train
from sample import Sampler
from cat import CatProcess
from process import Process
from dataloader import mnist_dataset
from utils import InfoLogger, save_path

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    # add exp name
    parser.add_argument('--exp', type=str, default='dev', help='experiment name')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--proc_name', type=str, default='simplex', help='process name')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--k', type=int, default=2, help='number of categories')
    parser.add_argument('--O', type=int, default=3, help='process origin')
    parser.add_argument('--h', type=int, default=10, help='process speed')
    parser.add_argument('--T', type=float, nargs='+', default=[0.08, 0.55], help='time interval')
    parser.add_argument('--a', type=int, default=10, help='simplex domain')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # asserts
    assert args.k > 1, 'k must be greater than 1'
    assert args.O > 0, 'O must be greater than 0'
    assert args.h > 0, 'h must be greater than 0'
    assert args.T[0] < args.T[1], 'T[0] must be less than T[1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'
    assert args.proc_name in ['cat', 'simplex'], 'process name must be cat or simplex'

    return args

# sample arguments as json
def save_args(args):
    # save args as json
    args_dict = vars(args)
    args_dict['T'] = list(args_dict['T'])
    os.makedirs(args.exp, exist_ok=True)
    with open(save_path(args, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

if __name__ == '__main__':
    args = get_args()
    args.exp = os.path.join('results', args.exp)
    save_args(args)
    print(f'device: {args.device}')

    # load dataset, model, optimizer and process
    loader = mnist_dataset(args.batch_size, args.k)
    ch = args.k if args.proc_name == 'cat' else args.k-1
    model = Unet(dim=64, channels=ch).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    #sampler = Sampler(args)
    logger = InfoLogger()

    # get process
    if args.proc_name == 'cat':
        T = 1000
        betas = torch.linspace(1e-4, 0.02, T)
        process = CatProcess(args.k, T, betas, args.device)
    elif args.proc_name == 'simplex':
        process = Process(args)

    # train loop
    loss_track = []
    for epoch in range(args.epochs):
        logger.clear()
        if args.proc_name == 'simplex':
            loss = train(model, process, loader, opt, logger, args)
        elif args.proc_name == 'cat':
            loss = cat_train(model, process, loader, opt, args)

        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # sample
        #sampler()

        # save model
        if epoch % 10 == 0:
            sp = save_path(args, f'model_{args.proc_name}_{epoch}.pt')
            torch.save(model.state_dict(), sp)

        # plot loss
        plt.plot(loss_track)
        plt.savefig(save_path(args, 'loss.png'))
        plt.close()

