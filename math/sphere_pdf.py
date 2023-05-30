import sys, os
import torch 
import numpy as np 

import matplotlib.pyplot as plt
plt.style.use('seaborn')

def f_inv(x):
    return x.norm(dim=1).log() - (1 - x.norm(dim=1)).log()

def f(x):
    return torch.sigmoid(x.norm(dim=1))

def pdf(x):
    d = x
    c = (2*np.pi) ** (d/2)

    a = f_inv(x)
    print(a)
    out = (-0.5 * a.square()).exp()
    return 1/c * out

if __name__ == '__main__':
    n = 1000
    d = 8
    x = torch.randn(n, d)

    #cv = lambda x: 1 / (2 * x.norm(dim=1).square() * (1 - x.norm(dim=1)).square())
    y = pdf(x)
    y = y.detach().numpy()

    plt.hist(y)
    plt.show()


