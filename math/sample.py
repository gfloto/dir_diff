import sys
import numpy as np
import matplotlib.pyplot as plt

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# logit function
def logit(x):
    return np.log(x / (1-x))

# pdf of logit normal distribution
def pdf(x, mu, v):
    return 1/np.sqrt(2*np.pi*v) * np.exp(-0.5/v * (logit(x) - mu)**2) / (x * (1-x))

'''
Check sampling from logit normal distribution
To sample, simply sample from a normal distribution and apply the sigmoid function
'''

if __name__ =='__main__':
    n = 10000

    # check sampling from logit normal distribution
    # random parameters
    mu = np.random.randn()
    v = np.random.rand()
    print(f'mu: {mu} v: {v}')

    # sample from logit normal distribution
    x = np.random.rand(n)
    y = sigmoid(np.sqrt(v) * np.random.randn(n) + mu)

    # plot histogram and pdf
    plt.hist(y, bins=100, density=True)

    x_ = np.linspace(0, 1, n+2)[1:-1]
    plt.plot(x_, pdf(x_, mu, v))
    plt.show()


