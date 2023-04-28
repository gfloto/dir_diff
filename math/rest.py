import sys, os
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.special import beta
plt.style.use('seaborn')

def beta_pdf(x, a, b):
    return x**(a-1) * (1-x)**(b-1) / beta(a, b)

'''
We can use this funky sampling scheme:
---------------
Sample n uniform points, then select the kth sorted point
This is the same as sampling from B(x;a,b)!
    - a = k+1, b = n-k

Now we only need f and g to write dx = f(x,t)dt + g(t)dw
'''

if __name__ == '__main__':
    # n total number of points
    # a1 = sorted point to select
    # a2 = n - a1
    N = 10000 # trials to get histogram
    n = 3 # sample n uniform points
    k = 1 # select the bth point

    # beta pdf parameters
    a = k+1
    b = n-k

    x = np.random.rand(N,n)
    x = np.sort(x, axis=1)
    y = x[:,k]
    
    bins = 20
    plt.hist(y, bins=bins, density=True)

    x_ = np.linspace(0,1,1000)
    plt.plot(x_, beta_pdf(x_, a, b))
    plt.show()

