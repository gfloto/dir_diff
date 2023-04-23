import sys, os
import torch
import numpy as np
import scipy
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from dataloader import mnist_dataset
from utils import a_logit, save_path
from model import Unet
from plot import save_vis, make_gif

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv_a(x) x(1-x/a) + 0.5 x(1-x/a)(1-2x/a)
g(x) = x(1-x/a)
'''

class Sampler:
    def __init__(self, args, batch_size, device):
        self.O = torch.tensor(args.O)
        self.h = torch.tensor(args.h)
        self.a = torch.tensor(args.a)
        self.t_min, self.t_max = args.T
        self.device = device

        self.batch_size = batch_size
        self.img_shape = (32, 32)

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*self.t_max))
        nu = torch.randn(self.batch_size, *self.img_shape)
        x = self.a * torch.sigmoid(var.sqrt() * nu)
        return x

    # drift term
    def f(self, x):
        a = self.a
        f1 = -self.h*a_logit(x,a) * x*(1 - x/a)
        f2 = 0.5*x*(1 - x/a) * (1 - 2*x/a)
        return f1 + f2 

    # diffusion term
    def g(self, x):
        return x*(1-x/self.a)

    # score at xt given mu and var
    def s(self, xt, mu, var):
        num = a_logit(xt, self.a) - 2*var*xt - mu + self.a*var
        denom = var*xt*(xt - self.a)
        score = num / denom
        return score
    
    def update_order(self, model, x, t, dt, order=1, g_scale=0.02):
        # get info for euler discretization of sde solution
        f = self.f(x)
        g = self.g(x)
        eps = torch.randn(self.batch_size, *self.img_shape).to(self.device) 
        score = model(x[:, None, ...], t).squeeze()

        # runge kutta solvers
        if order == 1:
            update = (f + g**2 * score)*dt + g_scale*g*eps 
        elif order == 2:
            k1 = dt * (f + g**2 * score) # k1 is same as euler first order
            x1 = x + k1
            f1 = self.f(x1)
            g1 = self.g(x1)
            t1 = t - dt

            score1 = model(x1[:, None, ...], t1).squeeze()
            k2 = dt * (f1 + g1**2 * score1)
            update = (k1+k2)/2 + g_scale*g1*eps 
        return update

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png', order=1, d=100, g_scale=0.02):
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)

        # select times for nn (always 0-1)
        dt = torch.tensor([1/T]).to(self.device)
        t = torch.tensor([1.]).to(self.device)

        # sample loop
        for i in range(T):
            if i % d == 0:
                print(f"Sample iteration {i}/{T}")
            update = self.update_order(model, x, t, dt, order=order, g_scale=g_scale)

            # update x
            x += update
            t -= dt

            # save sample
            if save_path is not None and (i % d == 0 or i == T-1):
                x_save = x / self.a # map back to prob. simplex
                save_vis(x_save, f'imgs/{int(i/d)}.png', k=None, x_out=update)

        # binarize
        x = x / self.a # map back to prob. simplex
        x = (x > 0.5).float()
        for i in range(int(T/d), int(T/d)+10):
            save_vis(x, f'imgs/{i}.png', k=None)

        # save gif
        if save_path is not None:
            make_gif('imgs', save_path, int(T/d)+10)
        return x

def sampler_wrapper(model, T, order, g_scale, batch_size=8):
    results = []
    batch_size, T = int(batch_size), int(T)
    order = 1 if order < 1.5 else 2
    for i in range(num_samples_run):
        sampler = Sampler(args, batch_size=batch_size, device='cuda')
        result = sampler(model, T, save_path=save_path(args, f'sample_{i}'), order=order, g_scale=g_scale)
        results.append(result)
    return results

def compute_fid(T, order, g_scale, batch_size=8):
    print(f'Computing FID for T={T}, order={order}, g_scale={g_scale}, batch_size={batch_size}')
    model_results = sampler_wrapper(model, T, order, g_scale, batch_size=batch_size)
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
    return 1-fid

import json
import argparse
if __name__ == '__main__':
    name = 'beta'
    path = os.path.join('results', name)

    # load json of args
    args = json.load(open(os.path.join(path, 'args.json'), 'r'))
    args = argparse.Namespace(**args)

    # load model
    model = Unet(dim=64, channels=1).to('cuda')
    model_path = os.path.join('results', name, f'model_{args.proc_name}_840.pt')
    model.load_state_dict(torch.load(model_path))
    model.eval() 

    # load data
    data = mnist_dataset(args.batch_size, args.k, num_workers=0, use_full_batch=True)
    true_data, _ = next(iter(data))
    true_data = true_data.reshape(-1, 32*32).cpu().numpy()
    mu_true, sigma_true = np.mean(true_data, axis=0), np.cov(true_data, rowvar=False)

    # Bayesian Optimization config
    num_samples_run = 1
    param_bounds = {'order': (1, 2), 'g_scale': (0.015, 0.032), 'batch_size': (8, 64), 'T': (1000, 3000)}

    bayes_optim = BayesianOptimization(
        f=compute_fid,
        pbounds=param_bounds,
        random_state=1,
        verbose=2,
    )
    logger = JSONLogger(path="./results/beta/bayes_optim_logs.json")
    bayes_optim.subscribe(Events.OPTIMIZATION_STEP, logger)
    bayes_optim.maximize(
        init_points=50,
        n_iter=180,
    )