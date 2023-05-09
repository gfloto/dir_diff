import sys, os
import torch
import math
import numpy as np
from einops import repeat, rearrange
from torch.nn.functional import one_hot

# useful torch -> numpy
def ptnp(x):
    return x.detach().cpu().numpy()

# append path to experiment folder
def save_path(args, path):
    return os.path.join(args.exp, path)

# [b, k, ...] to categorical [b, ...]
def onehot2cat(x, k):
    return torch.argmax(x, dim=1)

# [b, ...] to one-hot [b, k, ...]
def cat2onehot(x, k, shape):
    # shape is [b, ..., k]
    x = one_hot(x, k).float() 

    # TODO: there should be a better way to write this...
    if len(shape) == 3:
        return rearrange(x, 'b w k -> b k w')
    elif len(shape) == 4:
        return rearrange(x, 'b w h k -> b k h w')
    elif len(shape) == 5:
        return rearrange(x, 'b c w h k -> b k c h w')
    else:
        raise ValueError("shape not supported")

# useful function for returning an identity tensor
# if input is [b, k, ...] then it returns identity over i,j with [b, i, j] (i, j and k same size) 
def identity_tensor(x):
    # shape is [b, k, ...]
    shape = x.shape
    eye = torch.eye(x.shape[1])

    # this is ugly and bad!
    if len(shape) == 3:
        b, k, w = x.shape
        return repeat(eye, 'i j -> b i j w', b=b, w=w).to(x.device)
    elif len(shape) == 4:
        b, k, w, h = x.shape
        return repeat(eye, 'i j -> b i j w h', b=b, w=w, h=h).to(x.device)
    elif len(shape) == 5:
        b, k, c, w, h = x.shape
        return repeat(eye, 'i j -> b i j c w h', b=b, c=c, w=w, h=h).to(x.device)
    else:
        raise ValueError("shape not supported")

# TODO: eventually move this to metrics.py or smthn
def calculate_perplexity(model, data):
    """
    Calculates the perplexity of a language model on a given dataset.
    :param model: The language model to evaluate.
    :param data: The dataset to evaluate the model on.
    :return: The perplexity of the model of the dataset.
    """
    total_log_prob = 0
    num_words = 0
    for sentence in data:
        log_prob = model(sentence)
        total_log_prob += log_prob
        num_words += len(sentence)
    perplexity = math.exp(total_log_prob / num_words)
    return perplexity

# from t and loss data, get distribution to sample from
def sample_dist(t, loss, bins=50):
    total = np.zeros(bins)    
    count = np.zeros(bins)

    # get count for each bin
    assert len(t) == len(loss)
    for i in range(len(t)):
        bin_id = int(t[i] * bins)
        total[bin_id] += loss[i]
        count[bin_id] += 1

    # get average loss for each bin
    avg = np.zeros(bins)
    for i in range(bins):
        if count[i] > 0:
            avg[i] = total[i] / count[i]
        else:
            avg[i] = 0 

    # set 0 to min, then normalize
    avg = np.array(avg)
    avg[avg == 0] = np.min(avg[avg != 0])
    avg /= np.sum(avg)
    return avg

import matplotlib.pyplot as plt
if __name__ == '__main__':
    x = np.random.rand(400)
    y = np.exp(x)

    dist = sample_dist(x, y)
    out = np.random.choice(np.arange(dist.shape[0]), p=dist)
    print(out)
    sys.exit()

    # plot bar
    x_ = np.arange(len(dist)) / len(dist)
    plt.scatter(x_, dist)
    plt.show()