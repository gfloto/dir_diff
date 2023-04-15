import os, sys
import imageio  
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# logit function
def logit(x):
    return np.log(x / (1-x))

def sample(mu, var, n):
    return sigmoid(np.sqrt(var) * np.random.randn(n) + mu)

def mean_var(O, h, t):
    m = O * np.exp(-h*t)
    v = 1/(2*h) * (1 - np.exp(-2*h*t))
    return m, v

# score of logit normal distribution (d/dx log pdf)
def score(x, mu, v):
    num = logit(x) - 2*var*x - mu + var
    denom = var*x*(x-1)
    return num / denom

# pdf of logit normal distribution
def pdf(x, mu, v):
    return 1/np.sqrt(2*np.pi*v) * np.exp(-0.5/v * (logit(x) - mu)**2) / (x * (1-x))

if __name__ =='__main__':
    # process hyperparameters
    O = 6
    h = 4

    T = 100
    n = 10000
    # check score throughout process
    t = np.linspace(0, 1, T+2)[1:-1]

    # create directory for images
    import os
    if not os.path.exists('imgs'):
        os.mkdir('imgs')

    for i in tqdm(range(T)):
        # get mean and variance
        mu, var = mean_var(O, h, t[i])

        # sample n points
        x = sample(mu, var, n)

        # get score at each point (d/dx log pdf)
        sc = score(x, mu, var)

        # plot histogram and pdf
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        x_ = np.linspace(0, 1, n+2)[1:-1]
        ax1.hist(x, bins=100, density=True)
        ax1.plot(x_, pdf(x_, mu, var))
        ax1.set_title(f'PDF at t={t[i]:.2f}')

        ax2.hist(sc, bins=100, density=True)
        ax2.set_title(f'Score at t={t[i]:.2f}')

        # save figure
        path = f'imgs/plot_{i:03d}.png'
        plt.savefig(path)
        plt.close() 

    # create gif and delete images
    print('creating gif...')
    images = []
    for i in range(T):
        path = f'imgs/plot_{i:03d}.png'
        images.append(imageio.imread(path))
        os.remove(path)
    imageio.mimsave('process.gif', images)
    print('done')
