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
        b, k, w = x.shape
        return rearrange(x, 'b w k -> b k w', w=w)
    elif len(shape) == 4:
        b, k, w, h = x.shape
        return rearrange(x, 'b w h k -> b k h w', w=w, h=h)

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