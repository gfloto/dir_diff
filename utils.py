import torch

def onehot2cat(x, k):
    return torch.argmax(x, dim=1) / (k-1)