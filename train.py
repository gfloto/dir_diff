import sys, os
from tqdm import tqdm
import torch
import numpy as np
from torch.nn.functional import log_softmax, cross_entropy, kl_div
from einops import rearrange

from utils import ptnp, onehot2cat

def train(model, process, loader, opt, args):
    device = args.device; k = args.k
    model.train()
    loss_track = []
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

    return np.mean(loss_track)


from cat_utils import get_logits_from_logistic_pars
from cat_utils import kld_logits, cat_log_nll, vb_loss

def cat_train(model, process, loader, opt, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        if i > 10: break
        # difference in dataloaders (some output class info)
        if isinstance(x0, tuple) or (isinstance(x0, list) and len(x0)==2):
            x0 = x0[0]
        x0 = x0.to(args.device)

        # get t, x0 xt
        t, tu = process.t()
        xt = process.xt(x0, t) 

        # get model output
        pred = model(xt, tu)

        # option to output params of truncated logistic distribution
        if args.trunc_logistic:
            raise NotImplementedError

        # option to output p(x0 | xt) and get p(x_t | xt) from that
        if args.sparse_cat and t > 0:
            model_x0_logits = pred
            model_logits = process.Q_rev(model_x0_logits, xt, t)
        else:
            model_logits = pred

        # q(x_t-1 | xt, x0)
        q_rev = process.Q_rev(x0, xt, t)

        # option to do aux loss
        if args.lmbda is not None:
            loss_vb = vb_loss(x0, t, q_rev, model_logits)
            loss_aux = cat_log_nll(x0, model_x0_logits)
            loss = loss_vb + args.lmbda*loss_aux 
            sys.exit()
        else:
            loss = cat_log_nll(x0, model_x0_logits)

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        # plotting
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)