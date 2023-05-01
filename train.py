import sys, os
import torch
import numpy as np
from torch.distributions import Beta
from torch.nn.functional import kl_div as kld
from torch.nn.functional import log_softmax
from tqdm import tqdm

from utils import ptnp
from plot import save_vis

def train(model, process, loader, opt, logger, args):
    device = args.device; k = args.k

    model.train()
    loss_track = []
    for i, (x0, _) in enumerate(tqdm(loader)):
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


def get_logits_from_logistic_pars(loc, log_scale, num_classes):
    """Computes logits for an underlying logistic distribution."""
    
    # The loc and log_scale are assumed to be modeled for data re-scaled
    # such that the values {0, ...,K-1} map to the interval [-1, 1].
    # Shape of loc and log_scale: (batch_size, height, width, channels)
    loc = jnp.expand_dims(loc, axis=-1)
    log_scale = jnp.expand_dims(log_scale, axis=-1)
    
    # Shift log_scale such that if it’s zero the output distribution
    # has a reasonable variance.
    inv_scale = jnp.exp(- (log_scale - 2.))
    
    bin_width = 2. / (num_classes - 1.)
    bin_centers = jnp.linspace(start=-1., stop=1., num=num_classes,
         19 endpoint=True)
    bin_centers = jnp.expand_dims(bin_centers,
         21 axis=tuple(range(0, loc.ndim-1)))
    
    bin_centers = bin_centers - loc
    # Note that the edge bins corresponding to the values 0 and K-1
    # don’t get assigned all of the mass in the tails to +/- infinity.
    # So the logits correspond to unnormalized log probabilites of a
    # discretized truncated logistic distribution.
    log_cdf_min = jax.nn.log_sigmoid(inv_scale * (bin_centers - 0.5 * bin_width))
    log_cdf_plus = jax.nn.log_sigmoid(inv_scale * (bin_centers + 0.5 * bin_width))

    logits = log_minus_exp(log_cdf_plus, log_cdf_min)

    return logits


def log_minus_exp(a, b, epsilon=1.e-6):
    """Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable way."""
    return a + jnp.log1p(-jnp.exp(b - a) + epsilon)


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
        # we have L = L_vb + L_q
        pred = log_pred.exp()
        assert q_rev.sum(dim=1).allclose(torch.ones_like(q_rev.sum(dim=1)))
        assert pred.sum(dim=1).allclose(torch.ones_like(pred.sum(dim=1))) 
        
        # We have L_vb = L1 + L2 + L3 
        # Where 
        # L1 = \E_{q(x_0)}[D_{KL}[q(x_T|x_0)|p(x(T))]] 
        
        # L2 = sum_t=2^T \E_{q(xt|x0)}[D_{KL}[q(x_{t-1} | x_t, x_0)|p_\theta(x_{t-1} | x_t)]
        # L3 = -\E_{q(x_1|x_0)}[log p_\theta (x_0 | x_1)]
        # it is known that this can be expressed in terms of:
        loss = torch.sum(q_rev * (q_rev.log() - log_pred), dim=(1))
        # L_q is given by considering outputs of the q process
        # and the transition
        # with lambda = 0.001 here we use another nn
        # 

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
