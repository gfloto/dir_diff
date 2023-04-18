import sys, os
import torch
from tqdm import tqdm

from model import Unet
from plot import save_vis
from utils import make_gif

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv(x) x(1-x) + 0.5 x(1-x)(1-2x)
g(x) = x(1-x)
'''

class Sample:
    def __init__(self, O, h, t_min, t_max, batch_size, device):
        self.O = torch.tensor(O)
        self.h = torch.tensor(h)
        self.t_min = t_min
        self.t_max = t_max
        self.batch_size = batch_size
        self.img_shape = (32, 32)
        self.device = device

    # get intial distribution at t = t_max
    def init_sample(self):
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*self.t_max))
        x = torch.sigmoid(var.sqrt() * torch.randn(self.batch_size, *self.img_shape))
        return x

    # drift term
    def f(self, x):
        return -self.h*torch.logit(x) + 0.5*x*(1-x)*(1-2*x)

    # diffusion term
    def g(self, x):
        return x*(1-x)

    # score at xt given mu and var
    def s(self, xt, mu, var):
        num = torch.logit(xt) - 2*var*xt - mu + var
        denom = var*xt*(xt-1)
        score = num / denom
        return score

    # scale score by this factor
    def score_scale(self, t):
        mu = self.O * torch.exp(-self.h*t) 
        var = 1/(2*self.h) * (1 - torch.exp(-2*self.h*t))
        
        # get values at +- sigma
        b1 = torch.sigmoid(mu - torch.sqrt(var))
        b2 = torch.sigmoid(mu + torch.sqrt(var))

        # get score at +- sigma
        s1 = self.s(b1, mu, var)
        s2 = self.s(b2, mu, var)

        r = (s1.abs() + s2.abs()) / 2
        return r

    @torch.no_grad()
    def __call__(self, model, T, save_path='sample.png'):
        d = 10
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        # initialize sample
        x = self.init_sample().to(self.device)
        dt = (self.t_max - self.t_min) / T
        dt = torch.tensor([dt]).to(self.device)
        t = torch.tensor([self.t_max]).to(self.device)

        for i in tqdm(range(T)):
            # get info for euler discretization of sde solution
            f = self.f(x)
            g = self.g(x)
            r = self.score_scale(t).to(self.device)
            eps = torch.randn(self.batch_size, *self.img_shape).to(self.device) 

            # get score from model
            score = r * model(x[:, None, ...], t).squeeze()
            t -= dt

            # update x
            x += (f + g**2 * score)*dt + 0.025*g*eps 

            # save sample
            if save_path is not None:
                save_vis(x, f'imgs/{int(i/d)}.png', k=None)

        # binarize
        x = (x > 0.5).float()
        for i in range(int(T/d), int(T/d)+10):
            save_vis(x, f'imgs/{i}.png', k=None)

        # save gif
        if save_path is not None:
            make_gif('imgs', save_path, int(T/d)+10)

# sorts files alpha numerically (natural computer display)
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

if __name__ == '__main__':
    # get newest model from 'results/model{num}.pt'
    names = [f for f in os.listdir('results/') if 'model' in f]
    names = sorted(names, key=natural_key)
    model = Unet(dim=64, channels=1).to('cuda')
    model.load_state_dict(torch.load(f'results/{names[-1]}'))
    model.eval() 

    # print model name
    print(f'Using model: {names[-1]}')

    # sample from model
    sample = Sample(O=6, h=8, t_min=0.075, t_max=0.7, batch_size=8, device='cuda')
    sample(model, T=1000, save_path='results/sample')


