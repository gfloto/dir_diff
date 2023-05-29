import sys, os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from args import get_args
from model import Unet, Transformer
from train import train, cat_train
from dataloader import text8_dataset, mnist_dataset, cifar10_dataset
from cat import CatProcess
from cube_proc import CubeProcess
from process import Process
from utils import save_path, sample_dist
from plot import hist_plot

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
    print(f'method: {args.proc_type}, dataset: {args.dataset}')

    if args.proc_type in ['simplex', 'cube']:
        print(f'theta: {args.theta:.4f}, O: {args.O}, t-min: {args.t_min:.4f}, t-max: {args.t_max:.4f}')
    elif args.proc_type == 'cat':
        print(f'q method: {args.q_method}, k: {args.k}, T: {args.T}, sparse: {args.p_sparse}, trunc: {args.trunc_logistic}, sched_method: {args.sched_method}')

    # load dataset
    if args.dataset == 'text8':
        loader = text8_dataset(args)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args)
    elif args.dataset == 'cifar10':
        loader = cifar10_dataset(args)

    # load model and optimizer
    if args.dataset == 'mnist':
        ch = args.k if args.proc_type in ['cat', 'cube'] else args.k-1
        model = Unet(dim=64, channels=ch).to(args.device)
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    elif args.dataset == 'cifar10':
        ch = 3*args.k if args.proc_type in ['cat', 'cube'] else 3*(args.k-1)
        model = Unet(dim=128, channels=ch).to(args.device)
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    elif args.dataset == 'text8':
        model = Transformer(emb_dim=256, vocab_size=27).to(args.device)
        print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # check for pretrained model
    model_path = save_path(args, f'model.pt')
    if os.path.exists(model_path):
        # TODO: load optimizer and loss
        print('loading pretrained model')
        model.load_state_dict(torch.load(model_path))

    # load process
    if args.proc_type == 'cat':
        process = CatProcess(args)
    elif args.proc_type in 'simplex':
        process = Process(args)
        s_dist = None
    elif args.proc_type == 'cube':
        process = CubeProcess(args)

    # train loop
    loss_track, tu_track, batch_losses = [], [], []
    for epoch in range(args.epochs):
        if args.proc_type == 'simplex':
            loss_out, tu_out = train(model, process, loader, opt, args)
            loss = np.mean(loss_out)

            # plot histogram of tu, get dist. for sample
            hist_plot(tu_out, loss_out, save_path(args, f'hist.png'))

        elif args.proc_type == 'cat':
            loss = cat_train(model, process, loader, opt, args)

        elif args.proc_type == 'cube':
            loss_out, _ = train(model, process, loader, opt, args)
            loss = np.mean(loss_out)

        # keep track of loss for training curve
        loss_track.append(loss)
        print(f'epoch: {epoch}, loss: {loss}')

        # save model and optimizer if best loss
        if loss == min(loss_track):
            torch.save(model.state_dict(), save_path(args, f'model.pt'))
            torch.save(opt.state_dict(), save_path(args, f'opt.pt'))

        # plot loss 
        np.save(save_path(args, 'loss.npy'), np.array(loss_track))
        plt.plot(loss_track)
        plt.savefig(save_path(args, 'loss.png'))
        plt.close()

