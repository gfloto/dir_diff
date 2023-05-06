import sys, os
import torch
import json
import argparse
import matplotlib.pyplot as plt

from args import get_args
from model import Unet
from train import train, cat_train
from dataloader import text8_dataset, mnist_dataset
from cat import CatProcess
from process import Process
from utils import save_path

# sample arguments as json
def save_args(args):
    # save args as json
    args_dict = vars(args)
    os.makedirs(args.exp, exist_ok=True)
    with open(save_path(args, 'args.json'), 'w') as f:
        json.dump(args_dict, f)

if __name__ == '__main__':
    args = get_args()
    args.exp = os.path.join('results', args.exp)

    # save and print args
    save_args(args)
    print(f'device: {args.device}')
    
    if args.proc_type == 'simplex':
        print(f'theta: {args.theta:.4f}, O: {args.O}, t-min: {args.t_min:.4f}, t-max: {args.t_max:.4f}')
    elif args.proc_type == 'cat':
        print(f'q method: {args.q_method}, k: {args.k}, T: {args.T}, sparse: {args.p_sparse}, trunc: {args.trunc_logistic}')

    # load dataset
    if args.dataset == 'text8':
        loader = text8_dataset(batch_size=args.batch_size)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args.batch_size, args.k)
    elif args.dataset == 'cifar10':
        # TODO: get this working
        raise ValueError(f'not implemented yet: {args.dataset}')

    # load model and optimizer
    ch = args.k if args.proc_type == 'cat' else args.k-1
    model = Unet(dim=64, channels=ch).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load process
    if args.proc_type == 'cat':
        process = CatProcess(args)
    elif args.proc_type == 'simplex':
        process = Process(args)

    # train loop
    loss_track = []
    for epoch in range(args.epochs):
        if args.proc_type == 'simplex':
            loss = train(model, process, loader, opt, args)
        elif args.proc_type == 'cat':
            loss = cat_train(model, process, loader, opt, args)
    
        # keep track of loss for training curve
        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # save model if best loss
        if loss == min(loss_track):
            sp = save_path(args, f'model.pt')
            torch.save(model.state_dict(), sp)

        # plot loss
        plt.plot(loss_track)
        plt.savefig(save_path(args, 'loss.png'))
        plt.close()

