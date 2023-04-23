import os, sys
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
checking properties of generalized logit-gauss
    - is the score correct?
    - check via finite differences
'''

# batched matrix multipy
def bdot(a, b):
    B = a.shape[0]
    S = a.shape[1]
    return torch.bmm(a.view(B, 1, S), b.view(B, S, 1)).reshape(-1)

# pdf of logit-gauss
def p(x, mu, a=1, v=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # n = d-1
    n = x.shape[1] - 1

    # set mu to tensor
    mu = mu * torch.ones(x.shape[0], n)

    # normalizing constant for gauss
    z = np.power(2*np.pi, n/2)

    # change of variables component
    c1 = a / x.prod(dim=1)

    # sig_a_inv of normal
    x_ = x[:, :-1]
    xd = x[:, -1].unsqueeze(1)

    # g is [b, n]
    g = ((x_/xd).log() - mu)
    h = bdot(g, g) # batched dot product
    c2 = (-1/(2*v) * h).exp()

    p = c1 * c2 / z
    return p

# 2d plot of logit-gauss
if __name__ == '__main__':
    b = 5000
    # dist. params.
    d = 3; a = 10; v=0.1
    mu = np.random.rand()
    mu = 0

    # get point on simplex
    x = torch.rand(b, d)
    x /= x.sum(dim=1, keepdim=True)
    x *= a

    p = p(x, mu, a, v)

    # to numpy, x: [b, d-1], p: [b]
    p = p.numpy()
    x = x[:, :-1].numpy()

    if d == 2:
        # plot 1d
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.scatter(x, p, s=2)
        plt.show()

    if d == 3:
        # plot 2d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0, 3*np.std(p))

        ax.scatter(x[:, 0], x[:, 1], p, s=2)
        plt.show()