import argparse
import torch
from auto_params import auto_param

# helper to convert str to bool for argparse
def str2bool(x):
    if x in ['True', 'true']:
        return True
    elif x in ['False', 'false']:
        return False
    else:
        raise ValueError('x must be True or False')

# argparse
def get_args():
    parser = argparse.ArgumentParser()
    # add exp name
    parser.add_argument('--exp', type=str, default='general_mnist', help='experiment name')
    parser.add_argument('--k', type=int, default=3, help='number of categories')
    parser.add_argument('--proc_type', type=str, default='cat', help='process type: simplex or cat')
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset: mnist, cifar10 or text8')

    # TODO: this is for the simplex and is confusing to use the word cat
    parser.add_argument('--cat_mag', type=float, default=0.9, help='value s.t. s=[cat_mag, c, c, ...]')
    parse.add_argument('--sparse_cat', type=str, default='False', help='use sparse method to get p(x_t | x_{t-1})')
    parse.add_argument('--trunc_logistic', type=str, default='False', help='whether to use truncated logistic during training')

    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=250, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # convert str to bool
    args.sparse_cat = str2bool(args.sparse_cat)
    args.trunc_logistic = str2bool(args.trunc_logistic)

    # asserts
    assert args.k > 1, 'k must be greater than 1'
    assert args.cat_mag > 0 and args.cat_mag < 1, 'cat_mag must be [0,1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'
    assert args.proc_type in ['cat', 'simplex'], 'process name must be cat or simplex'
    assert args.dataset in ['mnist', 'cifar10', 'text8'], 'dataset must be mnist, cifar10 or text8'

    # get process param for simplex diffusion
    if args.proc_type == 'simplex':
        args = auto_param(args) 

    # sparse cat always true if trunc logistic is used
    if args.trunc_logistic and not args.sparse_cat:
        args.sparse_cat = True
        print('overriding sparse_cat to True since trunc_logistic is True')

    return args
