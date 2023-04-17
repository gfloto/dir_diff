import sys, os
import torch
import numpy as np
from tqdm import tqdm

from utils import ptnp, scale_t
from plot import save_vis

def train(model, process, loader, time_sampler, optimizer, logger, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, (x0, _) in enumerate(tqdm(loader)):
        if i >= 100: break
        # get t, x0 xt
        x0 = x0.to(args.device)
        #t = torch.rand(1).to(device)
        t = time_sampler()
        xt = process.xt(x0, t)

        # get correct score and predicted score
        score = process.score(x0, xt, t)
        # NOTE: this is only for k=2
        score_out = model(xt[:, None, ...], t).squeeze()

        # loss
        loss = torch.mean((score_out - score)**2, dim=(1,2,))

        # store loss and time
        logger.store_loss(ptnp(loss.log()), ptnp(t))
        time_sampler.update(ptnp(loss.log()), ptnp(t))

        # backward pass
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        loss_track.append(ptnp(loss))

        # plotting
        if i % 25 == 0 and i > 0:
            #time_sampler.fit()
            #time_sampler.plot('time_f.png')
            logger.plot_loss('loss_hist.png')

    return np.mean(loss_track)
