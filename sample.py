import sys, os

import scipy
import torch
import numpy as np
from tqdm import tqdm
from einops import repeat, rearrange

from dataloader import cifar10_dataset, mnist_dataset, text8_dataset
from model import Unet
from process import sig, Process 
from plot import save_vis, make_gif
from utils import onehot2cat, save_path

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from ignite.metrics import FID, InceptionScore
from pytorch_fid.inception import InceptionV3
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn as nn

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

        # set t to tensor, then get score
        t = torch.tensor(t).to(self.device)
        t = torch.rand(1).to(device)
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
        t = np.linspace(0,1,T+1)[::-1]
        t = np.power(t, 1.5)
        dt = t[:-1] - t[1:]

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

        x = onehot2cat(x, args.k)
        if args.dataset == 'mnist':
            x = repeat(x, 'b w h -> b c w h', c=3)
        return x.to(dtype=torch.float32)

class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y

def sampler_wrapper(model, T, order, g_scale_alpha, g_scale_beta, batch_size=8):
    batch_size, T = int(batch_size), int(T)
    order = 1 if order < 1.5 else 2
    # sample from model
    sampler = Sampler(args, batch_size=args.batch_size, device=device, fast=True)
    result = sampler(model, T=T, g_scale_beta=g_scale_beta, g_scale_alpha=g_scale_alpha, save_path=save_path(args, 'sample.gif'))
    return result

def compute_fid(T, g_scale_alpha, g_scale_beta, order=1, batch_size=8):
    print(f'Computing FID for T={T}, order={order}, g_scale_alpha={g_scale_alpha}, g_scale_beta={g_scale_beta}, batch_size={batch_size}')
    model_results = sampler_wrapper(model, T, order, g_scale_alpha=g_scale_alpha, g_scale_beta=g_scale_beta, batch_size=batch_size)
    return compute_fid_score(model_results)

def compute_fid_score(model_results):
    # comparable metric
    pytorch_fid_metric = FID(num_features=dims, feature_extractor=wrapper_model)

    loader_batch = next(iter(loader))
    real_batch = interpolate(loader_batch[0].to(fid_device))
     # Convert model_results to tensor and move to device
    model_results = interpolate(torch.tensor(model_results).to(fid_device))
    
    # Update the FID metric with the generated data
    pytorch_fid_metric.update((model_results, real_batch))
    
    # Compute the FID score
    fid = pytorch_fid_metric.compute()

    print(f"fid score: {fid}")
    return -fid

def interpolate(batch):
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)

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

    # load dataset for torch ignite metrics
    dataset = args.dataset
    if args.dataset == 'text8':
        loader = text8_dataset(args.batch_size)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args.batch_size, args.k, create_onehot=False)
    elif args.dataset == 'cifar10':
        loader = cifar10_dataset(args.batch_size, args.k, create_onehot=False)

    # Bayesian Optimization config
    num_samples_run = 1
    param_bounds = {'g_scale_alpha': (0.01, 4), 'g_scale_beta': (1, 4), 'T': (1, 2)}

    # use cpu rather than cuda to get comparable results
    fid_device = "cpu"

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inception_model = InceptionV3([block_idx]).to(fid_device)

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(inception_model)
    wrapper_model.eval()

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