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
    parser.add_argument('--exp', type=str, default='city', help='experiment name')
    parser.add_argument('--k', type=int, default=10, help='number of categories')
    parser.add_argument('--proc_type', type=str, default='cube', help='process type: simplex or cat')
    parser.add_argument('--dataset', type=str, default='city', help='dataset: mnist, cifar10 or text8')
    
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    # simplex diffusion params
    parser.add_argument('--simplex_loc', type=float, default=0.9, help='value s.t. s=[simplex_loc, c, c, ...]')

    # categorical diffusion params
    parser.add_argument('--q_method', type=str, default='uniform', help='noising method for categorical')
    parser.add_argument('--sched_method', type=str, default='linear', help='noise scheduler method')
    # NOTE: DO NOT CHANGE lmbda
    parser.add_argument('--lmbda', type=float, default=None, help='loss factor when using truncated logistic training')
    parser.add_argument('--T', type=int, default=1000, help='time steps for discretizing time in [0,1]')
    parser.add_argument('--p_sparse', type=str, default='True', help='use sparse method to get p(x_t | x_{t-1})')
    parser.add_argument('--trunc_logistic', type=str, default='False', help='whether to use truncated logistic during training')

    args = parser.parse_args()

    # convert str to bool
    args.sparse_cat = str2bool(args.p_sparse)
    args.trunc_logistic = str2bool(args.trunc_logistic)

    # asserts
    assert args.exp is not None, 'must specify experiment name'
    assert args.k > 1, 'k must be greater than 1'
    assert args.simplex_loc > 0 and args.simplex_loc < 1, 'simplex_loc must be [0,1]'
    assert args.device in ['cuda', 'cpu'], 'device must be cuda or cpu'
    assert args.proc_type in ['cat', 'simplex', 'cube'], 'process name must be cat or simplex'
    assert args.dataset in ['mnist', 'cifar10', 'text8', 'city'], 'dataset must be mnist, cifar10 or text8'
    assert args.q_method in ['uniform', 'sparse', 'absorbing', 'gauss', 'knn'], 'cat_method must be uniform, sparse, absorbing, gaussian or knn'
    assert args.sched_method in ['linear', 'cosine', 'mutual_info']

    # text8 automatically has 27 categories
    if args.dataset == 'text8':
        args.k = 27
    elif args.dataset == 'city':
        args.k = 34

    # get process param for simplex diffusion
    if args.proc_type == 'simplex':
        args = auto_param(args) 
    elif args.proc_type == 'cube':
        print('using cube defaults args')
        args.theta = 5
        args.O = 3
        args.t_min = 0.08
        args.t_max = 0.8

    # sparse cat always true if trunc logistic is used
    if args.trunc_logistic:
        if not args.sparse_cat:
            args.sparse_cat = True
            print('overriding sparse_cat to True since trunc_logistic is True')
        if not args.lmbda:
            args.lmbda = 0.1
            print('overriding lmbda to 0.1 since trunc_logistic is True')

    return args
