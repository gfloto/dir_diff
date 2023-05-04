import sys, os
import torch
import numpy as np
from torch.distributions import Beta
from torch.nn.functional import kl_div as kld
from torch.nn.functional import log_softmax
from tqdm import tqdm

from utils import ptnp
from plot import save_vis

from einops import repeat

def train(model, process, loader, opt, args):
    device = args.device; k = args.k
    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        # get t, x0 xt
        x0 = x0[0].to(args.device)
        t, tu = process.t() # get scaled and unscaled t
        xt, mu, var = process.xt(x0, t)

        # learn g^2 score instead of score 
        g2_score = process.g2_score(xt, mu, var)

        # predict g^2 score
        score_out = model(xt, tu)

        # loss
        loss = (score_out - g2_score).pow(2).mean()

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # save loss
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)

def cat_train(model, process, loader, opt, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, (x0, _) in enumerate(tqdm(loader)):
        # get t, x0 xt
        x0 = x0.to(args.device)
        t = torch.randint(1, process.T, (1,)).to(device)
        xt = process.xt(x0, t.item())

        # p(x_t-1 | xt) ∝ sum_x0 q(x_t-1 | xt, x0) p(x0 | xt)
        pred = model(xt, t / process.T)
        log_pred = log_softmax(pred, dim=1)

        # q(x_t-1 | xt, x0)
        q_rev = process.q_rev(x0, xt, t.item())

        # loss
        pred = log_pred.exp()
        assert q_rev.sum(dim=1).allclose(torch.ones_like(q_rev.sum(dim=1)))
        assert pred.sum(dim=1).allclose(torch.ones_like(pred.sum(dim=1))) 

        loss = torch.sum(q_rev * (q_rev.log() - log_pred), dim=(1))
        # print values where loss is negative
        #if not (loss >= 0).all():
        #    print(loss[loss < 0])
        loss = loss.mean()
        #loss = kld(log_pred, q_rev, reduction='none', log_target=False).sum(dim=(1,2,3,)).mean() 

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # plotting
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)