import os
import sys
import torch
import torch.utils.data as data
import torchvision
import requests
from torchvision import transforms
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from einops import rearrange

import numpy as np
from PIL import Image

from torch.functional import F

# convert to onehot encoding with k categories
class Onehot(object):
    def __init__(self, k=10):
        self.k = k

    def __call__(self, x):
        x *= self.k-1
        x = torch.round(x).squeeze().type(torch.int64)  # remove channel dim
        x = one_hot(x, num_classes=self.k).type(torch.float32)
        return rearrange(x, 'h w k -> k h w')

# return mnist dataset
def mnist_dataset(batch_size, k, root='data/', num_workers=4, size=32):
    gray_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=True),
        Onehot(k)])
    mnist_set = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=gray_transform)
    mnist_loader = data.DataLoader(
        mnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return mnist_loader

# text8 dataset
# data available at "http://mattmahoney.net/dc/text8.zip"
# based off of https://github.com/undercutspiky/Char_LM_PyTorch/blob/master/dataloader.py
class Text8Dataset(Dataset):
    def __init__(self, chunk_size=256):
        # to convert from character to index and back
        self.char2idx = {char: idx for idx, char in enumerate(
            ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '])}
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        self.chunk_size = chunk_size
        self.num_tok = len(self.char2idx)
        self.data = self.load_data()

    # download dataset if it doesn't exist, or load pt file
    def load_data(self):
        # if data exists, load data
        if os.path.exists('data/text8.pt'):
            print('Loading data...')
            return torch.load('data/text8.pt')

        # check if data exists otherwise make GET request to data URL string
        else:
            if not os.path.exists('data'):
                print('Downloading data...')
                os.makedirs('data')

                url = 'http://mattmahoney.net/dc/text8.zip'
                r = requests.get(url, allow_redirects=True)
                open('data/text8.zip', 'wb').write(r.content)
                print('Download complete.')

            # unzip data
            print('Processing data...')
            import zipfile
            with zipfile.ZipFile('data/text8.zip', 'r') as zip_ref:
                zip_ref.extractall('data')

            # load data and split into chunks
            with open('data/text8', 'r') as f:
                data = f.read()
            chunks = [data[i:i+self.chunk_size] for i in range(0, len(data), self.chunk_size)]

            # tokenize and encode chunks
            all_tokens = []
            for i, chunk in enumerate(chunks):
                tokens = [self.char2idx[character] for character in list(chunk)]
                all_tokens.append(tokens)

            # save for faster loading 
            data = torch.tensor(all_tokens, dtype=torch.long)
            torch.save(data, 'data/text8.pt')
            return data

    def __getitem__(self, index):
        # load textual data
        text = self.data[index]
        out = F.one_hot(text, num_classes=self.num_tok).type(torch.float32)
        return rearrange(out, 'w k -> k w')

    def __len__(self):
        return self.data.shape[0]

# convenience
def text8_dataset(batch_size, num_workers=4, chunk_size=256):
    text8_set = Text8Dataset(chunk_size=chunk_size)
    text8_loader = data.DataLoader(
        text8_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return text8_loader

# TODO: check output of this...
if __name__ == "__main__":
    text8_test = Text8Dataset()
    print(text8_test[0], text8_test[0].shape)
    text8_test_loader = text8_dataset(32)
    for batch in text8_test_loader:
        print(batch.shape)
        break
