import os, sys
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
for plotting the pdf of logit-gauss in 1 and 2d
'''

# batched matrix multipy
def bdot(a, b):
    s1, s2 = a.shape
    return torch.bmm(a.view(s1, 1, s2), b.view(s1, s2, 1)).reshape(-1)

# pdf of logit-gauss
def p(x, mu, a=1, v=1):
    # ensure shapes are correct
    assert len(x.shape) == 2

    # x_ -> x
    xd = a - x.sum(dim=1, keepdim=True)
    x = torch.cat((x, xd), dim=1)

    # n = d-1
    n = x.shape[1] - 1

    # set mu to tensor
    mu = mu * torch.ones(x.shape[0], n)

    # normalizing constant for gauss
    z = np.power(2*np.pi, n/2)

    # change of variables component
    c1 = a / x.prod(dim=1)

    # seperate variables
    x_ = x[:, :-1]
    xd = x[:, -1].unsqueeze(1)

    # g is [b, n]
    g = (x_.log() - xd.log() - mu)
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
    x = x[:, :-1]

    out = p(x, mu, a, v)

    # to numpy, x: [b, d-1], p: [b]
    out = out.numpy()
    x = x.numpy()

    if d == 2:
        # plot 1d
        fig = plt.figure()
        ax = fig.add_subplot(111)

        plt.scatter(x, out, s=2)
        plt.show()

    if d == 3:
        # plot 2d
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(0, 3*np.std(out))

        ax.scatter(x[:, 0], x[:, 1], out, s=2)
        plt.show()