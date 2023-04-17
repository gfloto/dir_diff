import sys, os
import torch

from plot import save_vis

def train(model, process, loader, time_sampler, optimizer, args):
    device = args.device; k = args.k

    model.train()
    for i, (x, y) in enumerate(loader):
        # get t, x0 xt
        x0 = x.to(args.device)
        t = torch.rand(1).to(device)
        xt = process.xt(x0, t)

        # get correct score and predicted score
        score = process.score(x0, xt, t)
        score_out = model(x, t)

        # loss
        loss = torch.mean((score_out - score)**2)

        # backward pass
        optimizer.zero_grad()
        loss.backward()

        # show image
        path = 'test.png'
        x = save_vis(x, path, k)
        break