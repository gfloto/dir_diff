import sys, os

import scipy
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from dataloader import mnist_dataset
from model import Unet
from process import sig, Process 
from plot import save_vis, make_gif
from utils import save_path

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv_a(x) x(1-x/a) + 0.5 x(1-x/a)(1-2x/a)
g(x) = x(1-x/a)
'''

class Sampler:
    def __init__(self, args, batch_size, device, fast=False):
        self.O = args.O
        self.t_min = args.t_min
        self.t_max = args.t_max
        self.theta = args.theta 

        self.k = args.k
        self.device = device
        self.batch_size = batch_size
        self.process = Process(args) # for sde_f and sde_g

        # shapes for image vs text datasets
        if args.dataset in ['mnist', 'cifar10']:
            self.end_shape = (32, 32)
        elif args.dataset == 'text8':
            self.end_shape = (8)

        # disable cpu <-> gpu copies and disable gif creation
        self.fast = fast

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.theta)
        if args.dataset == 'mnist':
            sample = (np.sqrt(var) * torch.randn(self.batch_size, self.k-1, *self.end_shape)).to(self.device)
        elif args.dataset == 'cifar10':
            sample = (np.sqrt(var) * torch.randn(self.batch_size, self.k-1, 3, *self.end_shape)).to(self.device)
        x = sig(sample) 
        return x

    # backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dB
    # TODO: re-impliment RK2
    def update(self, model, x, t, dt, g_scale=1):
        # get f, g, g^2 score and dB
        f = self.process.sde_f(x)
        g = self.process.sde_g(x)     

        g2_score = model(x, t)

        # check f is not nan
        assert torch.isnan(f).sum() == 0, f'f is nan: {f}'
        dB = (np.sqrt(dt) * torch.randn_like(g2_score)).to(self.device) 

        # solve sde: https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method 
        gdB = torch.einsum('b i j ..., b j ... -> b i ...', g, dB)
        return (-f + g2_score)*dt + g_scale*gdB

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png', order=1, d=10, pad=1e-3, g_scale_alpha = 1.75, g_scale_beta=1.5):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # select times for nn (always 0-1)
        t = torch.tensor([1.]).to(self.device)
        # dt for sde solver
        t_norm = args.t_max - args.t_min
        dt = t_norm * 1/T

        # noise schedule
        g_scale = np.linspace(0,1,T)[::-1]
        g_scale = g_scale_alpha*np.power(g_scale, g_scale_beta)

        # time schedule
        cdf_t = np.linspace(0,1,T+1)[::-1]
        t = np.power(cdf_t, 1.5)
        dt = cdf_t[1:] - cdf_t[:-1]

        # sample loop
        d = 20
        for i in tqdm(range(T)):
            # update x
            change = self.update(model, x, t[i], dt[i], g_scale[i])
            x = x + change

            # keep in simplex 
            x = torch.clamp(x, pad, 1-pad)
            xsum = x.sum(1, keepdim=True)
            x = torch.where(xsum > 1-pad, (1-pad)*x/xsum, x)

            # save sample
            if save_path is not None and not self.fast:
                # normalize change to be [0, 1] to visualize 
                change = (change - change.min()) / (change.max() - change.min())
                save_vis([x.clone(), change], f'imgs/{int(i/d)}.png', k=self.k)
        if not self.fast:
            # discretize
            for i in range(int(T/d), int(T/d)+10):
                save_vis(x, f'imgs/{i}.png', k=self.k)

            # save gif
            if save_path is not None:
                make_gif('imgs', save_path, int(T/d)+10)

        return x

def sampler_wrapper(model, T, order, g_scale_alpha, g_scale_beta, batch_size=8):
    results = []
    batch_size, T = int(batch_size), int(T)
    order = 1 if order < 1.5 else 2
    for i in range(num_samples_run):
        # sample from model
        sampler = Sampler(args, batch_size=256, device=device, fast=True)
        result = sampler(model, T=T, g_scale_beta=g_scale_beta, g_scale_alpha=g_scale_alpha, save_path=save_path(args, 'sample.gif'))
        results.append(result)
    return results

def compute_fid(T, g_scale_alpha, g_scale_beta, order=1, batch_size=8):
    print(f'Computing FID for T={T}, order={order}, g_scale_alpha={g_scale_alpha}, g_scale_beta={g_scale_beta}, batch_size={batch_size}')
    model_results = sampler_wrapper(model, T, order, g_scale_alpha=g_scale_alpha, g_scale_beta=g_scale_beta, batch_size=batch_size)
    model_results = torch.cat(model_results, dim=0)
    return compute_fid_score(model_results)

def compute_fid_score(model_results):
    model_results = model_results.reshape(-1, 32*32).cpu().numpy()
    mu_model, sigma_model = np.mean(model_results, axis=0), np.cov(model_results, rowvar=False)

    # compute fid
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    diff = mu_true - mu_model
    covmean, _ = scipy.linalg.sqrtm(sigma_true.dot(sigma_model), disp=False)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        msg = f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates'
        print(msg)
        offset = np.eye(sigma_true.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma_true + offset).dot(sigma_model + offset))
    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma_true) + np.trace(sigma_model) - 2 * tr_covmean

    print(f"fid score: {fid}")
    return -fid


import json
import argparse

# to load experiment
def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default=None, help='experiment name')
    args = parser.parse_args()

    assert args.exp is not None, 'Must specify experiment name'
    return args

def load_sample_data(dataset):
    if dataset == "mnist":
        data = mnist_dataset(args.batch_size, args.k, num_workers=0)
        true_data, _ = next(iter(data))
        true_data = true_data.reshape(-1, 32 * 32).cpu().numpy()
        mu_true, sigma_true = np.mean(true_data, axis=0), np.cov(true_data, rowvar=False)
    return mu_true, sigma_true

if __name__ == '__main__':
    batch_size = 128
    sample_args = get_sample_args()
    path = os.path.join('results', sample_args.exp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    if args.dataset == 'mnist':
        ch = args.k if args.proc_type == 'cat' else args.k-1
        model = Unet(dim=64, channels=ch).to(args.device)
    elif args.dataset == 'cifar10':
        ch = 3*args.k if args.proc_type == 'cat' else 3*(args.k-1)
        model = Unet(dim=64, channels=ch).to(args.device)

    model_path = os.path.join('results', sample_args.exp, f'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 
    print(f'Loaded model from {model_path}')

    # load dataset to compute mu_true, sigma_true for FID
    dataset = args.dataset # should be the name of the dataset
    mu_true, sigma_true = load_sample_data(dataset)

    # Bayesian Optimization config
    num_samples_run = 1
    param_bounds = {'g_scale_alpha': (0.01, 4), 'g_scale_beta': (1, 4), 'T': (1000, 3000)}

    bayes_optim = BayesianOptimization(
        f=compute_fid,
        pbounds=param_bounds,
        random_state=1,
        verbose=2,
    )

    logger = JSONLogger(path=os.path.join('results', sample_args.exp, "bayes_optim_logs.json"))
    bayes_optim.subscribe(Events.OPTIMIZATION_STEP, logger)
    bayes_optim.maximize(
        init_points=50,
        n_iter=750,
    )