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

# sample from logit normal distribution
def sample(mu, var, n):
    return sigmoid(np.sqrt(var) * np.random.randn(n) + mu)

# mean and variance of process
def mean_var(O, h, t):
    m = O * np.exp(-h*t)
    v = 1/(2*h) * (1 - np.exp(-2*h*t))
    return m, v

# score of logit normal distribution (d/dx log pdf)
def score(x, mu, var):
    num = logit(x) - 2*var*x - mu + var
    denom = var*x*(x-1)
    return num / denom

# pdf of logit normal distribution
def pdf(x, mu, v):
    return 1/np.sqrt(2*np.pi*v) * np.exp(-0.5/v * (logit(x) - mu)**2) / (x * (1-x))

# reparam (s.t. the neural net isn't learning things in -5000 to 100 or something)
class Reparam:  
    # set s.t. the score for +- sigma = += 1 (linear transformation) 
    def __init__(self, O, h, T, sig=1):
        self.score_sig = np.zeros((2, T))
        self.r = np.zeros(T)
        for i in range(T):
            # get mean and variance
            mu, var = mean_var(O, h, t[i])           

            # get values at +- sigma
            b1 = sigmoid(mu - sig * np.sqrt(var))
            b2 = sigmoid(mu + sig * np.sqrt(var))

            # get score at +- sigma
            s1 = score(b1, mu, var)
            s2 = score(b2, mu, var)

            self.r[i] = (np.abs(s1) + np.abs(s2)) / 2

            # record score at +- sigma
            self.score_sig[0, i] = min(s1, s2)
            self.score_sig[1, i] = max(s1, s2)
    
    # reparam from score to norm
    def forward(self, x, t):
        return x / self.r[t]
        #return (x - self.score_sig[0,t]) / (self.score_sig[1,t] - self.score_sig[0,t])
    
    # reparam from norm to score
    def backward(self, x, t):
        return self.r[t] * x
        #return x * (self.score_sig[1,t] - self.score_sig[0,t]) + self.score_sig[0,t]

'''
Forward process and normalization for smoother nn learning
'''

if __name__ =='__main__':
    # process hyperparameters
    O = 6
    h = 8

    T = 100
    n = 10000
    # check score throughout process
    t = np.linspace(0.075, 0.7, T)

    # make reparam
    reparam = Reparam(O, h, T)

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
        sc_n = reparam.forward(sc, i)

        # plot histogram and pdf
        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        x_ = np.linspace(0, 1, n+2)[1:-1]
        ax1.hist(x, bins=100, density=True)
        ax1.plot(x_, pdf(x_, mu, var))
        ax1.set_title(f'PDF at t={t[i]:.2f}')

        ax2.hist(sc, bins=100, density=True)
        ax2.set_title(f'Score at t={t[i]:.2f}')

        ax3.scatter(x, sc, s=1)
        ax3.set_title(f'Score(x) at t={t[i]:.2f}')

        ax4.hist(sc_n, bins=100, density=True)
        ax4.set_title(f'Score after reparam at t={t[i]:.2f}')

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
