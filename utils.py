import sys, os
import torch
import math
import numpy as np
from einops import rearrange
from torch.nn.functional import one_hot

# append path to experiment folder
def save_path(args, path):
    return os.path.join(args.exp, path)

# [b, k, ...] to categorical [b, ...]
def onehot2cat(x, k):
    return torch.argmax(x, dim=1)

# [b, ...] to one-hot [b, k, ...]
def cat2onehot(x, k):
    x = one_hot(x, k).float() 
    return rearrange(x, 'b h w k -> b k h w')

# useful torch -> numpy
def ptnp(x):
    return x.detach().cpu().numpy()

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