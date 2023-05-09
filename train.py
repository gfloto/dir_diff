import sys, os
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.functional import softmax 

from utils import ptnp, onehot2cat

def train(model, process, loader, opt, args, track_tu = False):
    device = args.device; k = args.k
    model.train()
    loss_track, tu_track = [], []
    for i, x0 in enumerate(tqdm(loader)):
        # difference in dataloaders (some output class info)
        if isinstance(x0, tuple) or isinstance(x0, list):
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

        if track_tu: 
            tu_track.append(ptnp(tu))

    if track_tu: 
        return np.mean(loss_track), (tu_track, loss_track)
    
    return np.mean(loss_track)


from cat_utils import get_logits_from_logistic_pars
from cat_utils import kld_logits, cat_log_nll, vb_loss

def cat_train(model, process, loader, opt, args):
    # option to output params of truncated logistic distribution
    if args.trunc_logistic or not args.sparse_cat:
        raise NotImplementedError

    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        # difference in dataloaders (some output class info)
        if isinstance(x0, tuple) or (isinstance(x0, list) and len(x0)==2):
            x0 = x0[0]
        x0 = x0.to(args.device)

        # get t, x0, xt, q(x_t-1 | xt, x0)
        t, tu = process.t()
        xt = process.xt(x0, t) 
        q_rev_logits = process.Q_rev_logits(x0, xt, t)

        # get model output
        model_x0_logits = model(xt, tu)

        # output p(x0 | xt) and get p(x_t | xt) from that
        if t > 0:
            model_logits = process.Q_rev_logits(softmax(model_x0_logits, dim=1), xt, t)
        else:
            model_logits = model_x0_logits

        # losses
        loss_vb = vb_loss(x0, t, q_rev_logits, model_logits)
        if args.lmbda is not None:
            loss_aux = cat_log_nll(x0, model_x0_logits)
            loss = loss_vb + args.lmbda*loss_aux 
        else:
            loss = vb_loss(x0, t, q_rev_logits, model_logits)

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # plotting
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)