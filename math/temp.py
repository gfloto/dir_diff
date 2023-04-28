import sys
import torch
import numpy as np

# initial point for OU process
def get_O(x, a=1, f=0.95):
    s = (1-f)/(x.shape[0]-1)
    f1 = f - s
    x = f1*x + s

    print(x)

if __name__ == '__main__':
    a = 1
    d = 4

    # make vector on simplex
    x = torch.zeros(d)
    ind = np.random.randint(d)
    x[ind] = 1

    # initial point for OU process
    O = get_O(x, a)
