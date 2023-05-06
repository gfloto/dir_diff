import sys, os
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.functional import log_softmax, cross_entropy

from utils import ptnp

def train(model, process, loader, opt, args):
    device = args.device; k = args.k
    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        # difference in dataloaders (some output class info)
        if isinstance(x0, tuple):
            x0 = x0[0] 
        x0 = x0.to(args.device)

        # get t, x0 xt
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


from cat_utils import get_logits_from_logistic_pars

def cat_train(model, process, loader, opt, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        # difference in dataloaders (some output class info)
        if isinstance(x0, tuple):
            x0 = x0[0] 
        x0 = x0.to(args.device)

        # get t, x0 xt
        t, tu = process.t()
        xt = process.xt(x0, t) 

        # get model output
        pred = model(xt, tu)

        # option to output params of truncated logistic distribution
        if args.trunc_logistic:
            loc, log_scale = pred
            # TODO: are these normalized??
            logits = get_logits_from_logistic_pars(loc, log_scale, args.k)
        else:
            # TODO: some missing detail here!!
            logits = log_softmax(pred, dim=1)

        # option to output p(x0 | xt) and get p(x_t | xt) from that
        if args.sparse_cat:
            logits = process.Q_rev(logits, xt, t)
            
        # q(x_t-1 | xt, x0)
        q_rev = process.Q_rev(x0, xt, t)

        # TODO: use torch kld; logits must be normalized... 
        loss_vb = (q_rev * (q_rev.log() - logits)).sum(1)

        # option to do aux loss
        if args.lmbda is not None:
            loss_aux = cross_entropy(logits, onehot_to_int(q_rev)).sum(1)
            loss = loss_vb + args.lmda*loss_aux 
        else:
            loss = loss_vb 

        # backward pass
        loss = loss.mean() 
        opt.zero_grad()
        loss.backward()
        opt.step()

        # plotting
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)