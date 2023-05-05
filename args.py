import argparse
import torch
from auto_params import auto_param

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    # add exp name
    parser.add_argument('--exp', type=str, default='general_mnist', help='experiment name')
    parser.add_argument('--k', type=int, default=3, help='number of categories')
    parser.add_argument('--proc_type', type=str, default='simplex', help='process type: simplex or cat')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset: mnist, cifar10 or text8')
    parser.add_argument('--cat_mag', type=float, default=0.9, help='value s.t. s=[cat_mag, c, c, ...]')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # asserts
    assert args.k > 1, 'k must be greater than 1'
    assert args.cat_mag > 0 and args.cat_mag < 1, 'cat_mag must be [0,1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'
    assert args.proc_type in ['cat', 'simplex'], 'process name must be cat or simplex'
    assert args.dataset in ['mnist', 'cifar10', 'text8'], 'dataset must be mnist, cifar10 or text8'

    # get process param for simplex diffusion
    if args.proc_type == 'simplex':
        args = auto_param(args) 

    return args
