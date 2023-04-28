import os, sys
import torch
from torch.func import jacrev, vmap
import numpy as np
from einops import repeat

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
checking properties of generalized logit-gauss
    - is the score correct?
    - check via finite differences
'''

# batched matrix multipy
def bdot(a, b):
    s1, s2 = a.shape
    return torch.bmm(a.view(s1, 1, s2), b.view(s1, s2, 1)).reshape(-1)

# log pdf of logit-gauss
def log_p(x, mu=0, a=10, v=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # x_ -> x
    xd = a - x.sum(dim=1, keepdim=True)
    x = torch.cat((x, xd), dim=1)

    # n = d-1
    n = x.shape[1] - 1

    # set mu to tensor
    mu = mu * torch.ones(x.shape[0], n)

    # change of variables component (this is for gradient, so the constant term can be forgotten)
    c1 = x.log().sum(dim=1)

    # seperate variables
    x_ = x[:, :-1]
    xd = x[:, -1].unsqueeze(1)

    # g is [b, n]
    g = (x_.log() - xd.log() - mu)
    h = bdot(g, g) # batched dot product
    c2 = 1/(2*v) * h

    log_p = -(c1 + c2)
    return log_p

# score or grad log pdf
def score(x, mu=0, a=10, v=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # x_ -> x
    xd = a - x.sum(dim=1, keepdim=True)
    x = torch.cat((x, xd), dim=1)

    # n = d-1
    n = x.shape[1] - 1

    # set mu to tensor
    mu = mu * torch.ones(x.shape[0], n)

    # sig_a_inv of normal
    x_ = x[:, :-1]
    xd = x[:, -1].unsqueeze(1)

    # constant factor
    c1 = 1/(xd.squeeze()) * (x_.log() - xd.log() - mu).sum(dim=1)
    c1 = repeat(c1, 'b -> b k', k=n)

    # unique element-wise factor
    c2 = 1/(x_) * (x_.log() - xd.log() - mu)

    # change of variables component
    c3 = (x_ - xd) / (x_ * xd)

    score = -1/v * (c1 + c2) + c3
    return score


# 2d plot of logit-gauss
# TODO: I'm a coder noob and can't do partial application quickly...
if __name__ == '__main__':
    b = 5000
    # dist. params.
    d = 3; a=10; v=1

    # get point on simplex
    x = torch.rand(b, d)
    x /= x.sum(dim=1, keepdim=True)
    x *= a

    # remove x if any element in dim=1 is < c and > 1-c
    c = 0.1*a
    x = x[(x > c).all(dim=1) & (x < a-c).all(dim=1)]
    x = x[:, :-1]

    # get log_p and score
    lp = log_p(x)
    sc = score(x)

    score = score(x)
    glp = jacrev(log_p)(x) 
    glp = torch.diagonal(glp).T 

    # to numpy
    glp = -glp.detach().numpy()
    score = -score.detach().numpy()

    # average diff between score and glp
    print(f'avg diff: {np.mean(np.abs(score - glp))}, std: {np.std(np.abs(score - glp))}')
    print(f'avg rel diff: {np.mean(np.abs(score - glp)/np.abs(score))}, avg rel std: {np.std(np.abs(score - glp)/np.abs(score))}')

    if d != 3:
        sys.exit(0)

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    #ax1.set_zlim(0, 3*np.std(score[:, 0]))
    #ax2.set_zlim(0, 3*np.std(score[:, 1]))

    ax1.scatter(x[:, 0], x[:, 1], score[:, 0], s=2)
    ax1.scatter(x[:, 0], x[:, 1], glp[:, 0], s=2)

    ax2.scatter(x[:, 0], x[:, 1], score[:, 1], s=2)
    ax2.scatter(x[:, 0], x[:, 1], glp[:, 1], s=2)

    # legend
    ax2.legend(['score', 'finite diff'])
    plt.show()
