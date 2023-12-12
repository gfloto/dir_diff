import torch
from torch.func import jacrev
from einops import repeat
from functools import partial

# helper function
def gamma(x, mu):
    # x is [b, k, ...], mu is [k]
    b, k = x.shape[:2]
    mu = repeat(mu, 'k -> b k', b=b)
    if len(x.shape) > 2:
        c, w = x.shape[2:]
        mu = repeat(mu, 'b k -> b k c w', c=c, w=w)

    return x.log() - (1 - x.sum(dim=1, keepdim=True)).log() - mu

# load pdf of logit-gauss
# assumes that dim=1 is the simplex dimension
def log_p(x, mu, v):
    b, k = x.shape
    a1 = (x.prod(dim=1) * (1 - x.sum(dim=1)) ).log()

    # mu is [k], x is [b, k, h, w]
    a2 = gamma(x, mu).square().sum(dim=1)
    return - a1 - 1/(2*v) * a2

# score or grad log pdf
def score(x, mu, v):
    b, k, h, w = x.shape
    xd = 1 - x.sum(dim=1, keepdim=True)
    g = gamma(x, mu)

    a1 = (x - xd) / (x * xd)
    a2 = g / x
    a3 = g.sum(dim=1, keepdim=True) / xd

    return a1 - 1/v * (a2 + a3)

'''
this code checks that the analytical gradient of the log pdf is correct
'''

if __name__ == '__main__':
    # distribution parameters
    batch_size = 8
    d = 2
    h = 32
    w = 32

    # set mu and v
    mu = torch.randn(d-1)
    v = torch.tensor(1.)

    # get log pdf partially applied
    log_p = partial(log_p, mu=mu, v=v)

    # get a point on the simplex
    x = torch.rand(batch_size, d, h, w)
    x /= x.sum(dim=1, keepdim=True)
    x = x[:, :-1]

    # get random pixels
    i = torch.randint(h, size=(1,)).item()
    j = torch.randint(w, size=(1,)).item()
    x_ = x[..., i, j]

    # compute numerical gradient of log pdf
    num_grad = jacrev(log_p)(x_)
    num_grad = torch.diagonal(num_grad).T

    # get analytical gradient of log pdf
    ana_grad = score(x, mu, v)
    ana_grad = ana_grad[..., i, j]

    # compare
    print(num_grad)
    print(ana_grad)
    print(ana_grad.max(), ana_grad.min())