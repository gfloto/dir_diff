import sys, os
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from model import Unet
from plot import save_vis
from utils import make_gif, plot_score_norm, plot_score_std

'''
score matching sampling from: https://arxiv.org/abs/2011.13456
forward: dx = f(x,t)dt + g(t)dw
backward: dx = [f(x,t) - g(x)^2 score_t(x)]dt + g(t)dw'

f(x,t) = -h sig_inv(x) x(1-x) + 0.5 x(1-x)(1-2x)
g(x) = x(1-x)
'''

class Sample:
    def __init__(self, O, h, t_min, t_max, dt_max_small, dt_min_small, dt_dividing_constant, dt_scheduler_step, use_dt_scheduler, batch_size, device):
        self.O = torch.tensor(O)
        self.h = torch.tensor(h)
        self.t_min = t_min
        self.t_max = t_max
        self.dt_max_small = dt_max_small
        self.dt_min_small = dt_min_small
        self.dt_dividing_constant = dt_dividing_constant
        self.dt_scheduler_step = dt_scheduler_step
        self.use_dt_scheduler = use_dt_scheduler
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
    
    def compute_dt_bounds(self, dt_max, dt_min, decay_factor):
        dt_max_new = dt_max - decay_factor
        dt_min_new = dt_min + decay_factor
        return dt_max_new, dt_min_new

    # run the scheduler for dt after dt_scheduler_step steps
    # need to reduce the time step towards the end to avoid artifacting 
    def dt_scheduler(self, i, t, dt_max, dt_min, decay_factor, decay_factor_small):
        if i >= self.dt_scheduler_step:
            dt_max_new, dt_min_new = self.compute_dt_bounds(dt_max, dt_min, decay_factor)
            if dt_max_new <= dt_min_new or t < 0.3:
                dt_max_small, dt_min_small = self.compute_dt_bounds(dt_max, dt_min, decay_factor_small)
                if dt_max_small <= dt_min_small:
                    return self.dt_max_small, self.dt_min_small
        else:
            dt_max_new, dt_min_new = self.compute_dt_bounds(dt_max, dt_min, 0)
        return dt_max_new, dt_min_new

    def dt_function(self, dt_max, dt_min, T, t, score): 
        # print("t", t, "score", score)
        dt =  (dt_max - dt_min) / T
        multiplier = (1/score) * 280
        # return dt * multiplier
        return dt

    def update_order(self, x, t, dt, order=2):
        # get info for euler discretization of sde solution
        f = self.f(x)
        g = self.g(x)
        r = self.score_scale(t).to(self.device)
        eps = torch.randn(self.batch_size, *self.img_shape).to(self.device) 
        # get score from model
        score = r * model(x[:, None, ...], t).squeeze()
        if order == 1:
            update, score = (f + g**2 * score)*dt + 0.025*g*eps 
        elif order == 2:
            # runge kuuta 2nd order: ff_{n+1} = f{n} + (k1+k2)/2
            k1 = dt * (f + g**2 * score) # k1 is same as euler first order
            x1_temp = x + k1
            f1 = self.f(x1_temp)
            g1 = self.g(x1_temp)
            t1 = t - dt
            r1 = self.score_scale(t1).to(self.device)
            score1 = r1 * model(x[:, None, ...], t1).squeeze()
            k2 = dt * (f1 + g1**2 * score1)
            update = (k1+k2)/2 + 0.022*g1*eps 
        return update, score


    @torch.no_grad()
    def __call__(self, model, save_path='sample.png'):
        d = 10
        if save_path is not None:
            os.makedirs('imgs', exist_ok=True)

        t_vec = []
        score_vec = []

        # initialize sample
        x = self.init_sample().to(self.device)
        dt_max, dt_min = self.t_max, self.t_min
        dt = (dt_max - dt_min) / 200
        print("dt initial: ", dt)
        dt = torch.tensor([dt]).to(self.device)
        t = torch.tensor([self.t_max]).to(self.device)
        i = 0
        score_norm = 0
        while t >= self.t_min:
            update, score = self.update_order(x, t, dt)
            score_norm = torch.linalg.vector_norm(score)

            if i % 100 == 0:
                print(f"t: {t.item()}, dt: {dt}, i: {i}, score norm (fro):{score_norm.item()}")

            t_vec.append(t.item())
            score_vec.append(score)

            # run scheduler for dt
            if self.use_dt_scheduler:
                dt_max, dt_min = self.dt_scheduler(i, t, dt_max, dt_min, 0.00004, 0.08)
            dt = self.dt_function(dt_max, dt_min, self.dt_dividing_constant, i, score_norm)


            # update x
            x += update
            t -= dt
            i += 1

            # save sample
            if save_path is not None and (i % d == 0 or i == self.dt_dividing_constant-1 or (i+1)%d==0 or (i-1)%d==0):
                save_vis(x, f'imgs/{int(i/d)}.png', k=None, x_out = score)
            # if i > 900:
            #     print("i", i, "t", t.item(), "dt", dt, "score", score_norm.item(), "t_max_current", t_max_current, "t_min_current", t_min_current)
            if t <= 0.23: 
                break 
        print("Final statistics: t: ", t.item(), "dt: ", dt, "i: ", i, "score norm (fro):", score_norm.item())
        # binarize
        x = (x > 0.5).float()
        for j in range(int(i/d), int(i/d)+10):
            save_vis(x, f'imgs/{j}.png', k=None)

        # save gif
        if save_path is not None:
            make_gif('imgs', save_path, int(i/d)+10)
        print("score_vec", score_vec)
        plot_score_std(t_vec, score_vec, save_path='score_std.png')
        plot_score_norm(t_vec, score_vec, save_path='score_norm.png')

# sorts files alpha numerically (natural computer display)
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

if __name__ == '__main__':
    torch.set_float32_matmul_precision("medium")
    # get newest model from 'results/model{num}.pt'
    names = [f for f in os.listdir('results/') if 'model' in f]
    names = sorted(names, key=natural_key)
    model = Unet(dim=64, channels=1).to('cuda')
    model.load_state_dict(torch.load(f'results/{names[-1]}'))
    model.eval() 

    # print model name
    print(f'Using model: {names[-1]}')

    # sample from model
    sample = Sample(O=6, h=8, t_min=0.075, t_max=0.7,  dt_min_small = 0.2, dt_max_small = 0.5, dt_dividing_constant=1000, dt_scheduler_step=280, use_dt_scheduler = True, batch_size=8, device='cuda')
    sample(model, save_path='results/sample')

  
