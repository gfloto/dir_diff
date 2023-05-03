import sys, os
import torch
import numpy as np
from torch.distributions import Beta
from torch.nn.functional import kl_div as kld
from torch.nn.functional import log_softmax
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import torch
import torch.nn.functional as F

from utils import ptnp
from plot import save_vis

def train(model, process, loader, opt, logger, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, x0 in enumerate(tqdm(loader)):
        # get t, x0 xt
        x0 = x0.to(args.device)
        t, tu = process.t() # get scaled and unscaled t
        xt, mu, var = process.xt(x0, t)

        # get correct score and predicted score
        score = process.score(xt, mu, var)
        score_out = model(xt, tu)

        # loss
        loss = torch.mean((score_out - score)**2)

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()
    
        # save loss
        loss_track.append(ptnp(loss))

    return np.mean(loss_track)


def normalize(int_vctr, num_classes):
    return 2*(int_vctr/num_classes) - 1


def get_logits_from_logistic_pars(loc, log_scale, num_classes):
    """
    Computes logits for an underlying logistic distribution.
    Adopted from Discrete Diffusion.

    Args:
        loc (torch.Tensor): A tensor containing the location parameters of the logistic distribution.
                            Shape: (batch_size, height, width, channels)
        log_scale (torch.Tensor): A tensor containing the log scale parameters of the logistic distribution.
                                  Shape: (batch_size, height, width, channels)
        num_classes (int): Number of classes in the discrete distribution.

    Returns:
        torch.Tensor: A tensor containing the logits for the logistic distribution.
    """

    # The loc and log_scale are assumed to be modeled for data re-scaled
    # such that the values {0, ...,K-1} map to the interval [-1, 1].
    loc = loc.unsqueeze(-1)
    log_scale = log_scale.unsqueeze(-1)

    # Shift log_scale such that if it’s zero the output distribution
    # has a reasonable variance.
    inv_scale = torch.exp(-(log_scale - 2.))

    bin_width = 2. / (num_classes - 1.)
    bin_centers = torch.linspace(start=-1., stop=1., steps=num_classes)
    bin_centers = bin_centers.view(*([1] * (loc.ndim - 1)), -1)

    bin_centers = bin_centers - loc
    # Note that the edge bins corresponding to the values 0 and K-1
    # don’t get assigned all of the mass in the tails to +/- infinity.
    # So the logits correspond to unnormalized log probabilites of a
    # discretized truncated logistic distribution.
    log_cdf_min = torch.log(torch.sigmoid(inv_scale * (bin_centers - 0.5 * bin_width)))
    log_cdf_plus = torch.log(torch.sigmoid(inv_scale * (bin_centers + 0.5 * bin_width)))

    logits = log_minus_exp(log_cdf_plus, log_cdf_min)

    return logits


def log_minus_exp(a, b, epsilon=1.e-6):
    """
    Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable way.

    Args:
        a (torch.Tensor): A tensor containing the a values.
        b (torch.Tensor): A tensor containing the b values.
        epsilon (float, optional): A small value to ensure numerical stability. Defaults to 1.e-6.

    Returns:
        torch.Tensor: A tensor containing the log(exp(a) - exp(b)) values.
    """
    return a + torch.log1p(-torch.exp(b - a) + epsilon)


def cat_train(model, process, loader, opt, args, lmbda=0.01):
    device = args.device; k = args.k
    model.train()
    loss_track = []
    for i, (x0,_) in enumerate(tqdm(loader)):
        # 
        #
        # STEP 1 ; Train Score Function Model
        #
        # get t, x0 xt
        x0 = x0.to(args.device)
        t = torch.randint(1, process.T, (1,)).to(device)
        xt = process.xt(x0, t.item())

        # get correct score and predicted score
        pred = model(xt, t / process.T)
        log_pred = log_softmax(pred, dim=1)
        q_rev = process.q_rev(x0, xt, t.item())

        assert q_rev.sum(dim=1).allclose(torch.ones_like(q_rev.sum(dim=1)))
        assert pred.sum(dim=1).allclose(torch.ones_like(pred.sum(dim=1))) 
        
        # We have vb_loss = L1 + L2 + L3 
        # Where 
        # L1 = \E_{q(x_0)}[D_{KL}[ q(x_T|x_0) | p(x(T)) ]]       
        # L2 = sum_t=2^T \E_{q(xt|x0)}[D_{KL}[q(x_{t-1} | x_t, x_0)|p_\theta(x_{t-1} | x_t)]
        # L3 = -\E_{q(x_1|x_0)}[log p_\theta (x_0 | x_1)]
        # It is known that this can be expressed in terms of
        vb_loss = torch.sum(q_rev * (q_rev.log() - log_pred), dim=(1))
        
        # L_aux is given by considering outputs of the q process
        xhatt = process.xt(q_rev, t.item())
        #aux_out = aux_model(normalize(onehot_to_int(xhatt), xhatt.shape[-1]), t.item())
        q_rev_hat = process.q_rev(q_rev, xhatt, t.item())
        aux_loss = cross_entropy(q_rev_hat, onehot_to_int(q_rev))
        # TODO move to cat.py
        #torch.sum(q_rev_hat * (q_rev_hat.log() - q_rev.log()), dim=(1))
        # logistic_params = get_logistic_params(aux_out)
        # log_scale = aux_out[..., :num_classes]
        # muprime = aux_out[..., num_classes:]
        # loc = torch.tanh(normalize(onehot_to_int(xhatt)) + muprime)
        # logits = get_logits_from_logistic_pars(loc, log_scale, num_classes)
        # aux_loss = cross_entropy(logits, onehot_to_int(q_rev))

        loss = loss.mean() + lmbda*aux_loss

        # backward pass
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_track.append(ptnp(loss))

    return np.mean(loss_track)
