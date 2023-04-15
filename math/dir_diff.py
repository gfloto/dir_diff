import sys, os
import numpy as np
import matplotlib.pyplot as plt

def dir_diff(x, dt=0.01):
    sig = 1
    kap = 0.01
    gam = 0.5

    # add drift term
    f = np.zeros_like(x)
    for i in range(x.shape[0]-1):
        f[i] = sig/2 * (gam*x[-1] - (1-gam)*x[i])
    x += f*dt

    # add diffusion term
    b = np.zeros_like(x)
    w = np.random.randn(x.shape[0])

    for i in range(x.shape[0]-1):
        b[i] = kap * x[i]*x[-1]
    d = np.sqrt(b)*w
    x += d*dt

    x[-1] = 1 - np.sum(x[:-1])
    return x

if __name__ == '__main__':
    d = 3
    # drift part of dirichlet diffusion
    x = np.random.rand(d)
    x /= np.sum(x)
    x[0] = 1; x[1] = 0; x[2] = 0
    print(x, np.sum(x))

    T = 1000
    store = np.zeros((T, d))
    for t in range(T):
        store[t] = x
        x = dir_diff(x)

    # plot store
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # make legend
    for i in range(d):
        ax.plot(store[:,i], label='x_%d' % i)
    ax.legend()
    plt.show()