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
from utils import InfoLogger, save_path, get_args

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

