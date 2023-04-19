import sys, os
import torch
import numpy as np
from torch.nn.functional import kl_div as kld
from torch.nn.functional import log_softmax
from tqdm import tqdm

from utils import ptnp, scale_t
from plot import save_vis

def train(model, process, loader, time_sampler, opt, logger, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, (x0, _) in enumerate(tqdm(loader)):
        # get t, x0 xt
        x0 = x0.to(args.device)
        #t = torch.rand(1).to(device)
        t = time_sampler()
        xt = process.xt(x0, t)

        # get correct score and predicted score
        score = process.score(x0, xt, t)
        score_out = model(xt[:, None, ...], t).squeeze()

        # loss
        loss = torch.mean((score_out - score)**2, dim=(1,2,))
        cl = loss.clone()
        loss = loss.mean()

        # store loss and time
        logger.store_loss(ptnp(cl.log()), ptnp(t))
        time_sampler.update(ptnp(cl.log()), ptnp(t))

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # plotting
        loss_track.append(ptnp(loss))
        #if i % 25 == 0 and i > 0:
            #time_sampler.fit()
            #time_sampler.plot('time_f.png')
            #logger.plot_loss('results/loss_hist.png')
            #save_vis(x0, 'results/noise.png', n=8, k=k, x_out=xt)

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

        # get correct score and predicted score
        pred = model(xt, t / process.T)
        log_pred = log_softmax(pred, dim=1)
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