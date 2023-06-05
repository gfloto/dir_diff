import os
import sys
import requests
import numpy as np
from PIL import Image
from einops import rearrange

import torch
import torchvision
from torchvision import transforms
from torch.functional import F
from torch.nn.functional import one_hot
import torch.utils.data as data
from torch.utils.data import Dataset

# convert categorical to binary along some axis
def cat2bin(x, k, axis=-1):
    # k in the number of categories
    # thus we has ceil(log_2(k)) bits
    len_ = int(np.ceil(np.log2(k)))

# general discretization class
class Discretize(object):
    def __init__(self, k):
        self.k = k

    def __call__(self, x):
        x *= self.k-1
        x = torch.round(x).type(torch.int64)
        x = one_hot(x, num_classes=self.k).type(torch.float32)
        x = rearrange(x, 'c h w k -> k c h w')
        return x.squeeze() # remove channel dim for mnist

# return mnist dataset
def mnist_dataset(args, root='data/', num_workers=4, size=32):
    gray_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size), antialias=True),
        Discretize(args.k)])
    mnist_set = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=gray_transform)
    mnist_loader = data.DataLoader(
        mnist_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    return mnist_loader

# return cifar 10 dataset
def cifar10_dataset(args, root='data/', num_workers=4, size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        Discretize(args.k)
    ])
    cifar10_set = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform)
    cifar10_loader = data.DataLoader(
        cifar10_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    return cifar10_loader

# return cifar 10 dataset
def city_dataset(args, num_workers=4, size=32):
    transform = transforms.Compose([
        transforms.Resize((2*size, 4*size), antialias=True),
        transforms.RandomCrop((size, 2*size)),
        transforms.RandomHorizontalFlip(),
        Discretize(args.k)
    ])
    city_set = CityDataset(
        img_path='data/cityscapes/labels', transform=transform)
    cifar10_loader = data.DataLoader(
        city_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    return cifar10_loader

# custom dataloader for city
class CityDataset(Dataset):
    def __init__(self, img_path, transform):
        self.img_path = img_path
        self.transform = transform

        # get all image names
        self.img_names = os.listdir(img_path)
        self.img_names = [name for name in self.img_names if name.endswith('.pt')]
        self.len_ = len(self.img_names)

    def __getitem__(self, index):
        # get images, then label
        img = torch.load(os.path.join(self.img_path, self.img_names[index]))
        img = img[None, ...]
        img = img.type(torch.float32)
        img /= 34.0

        # try using float16
        img = self.transform(img)
        return img

    def __len__(self):
        return self.len_

# return text 8 dataset
def text8_dataset(args, num_workers=4, chunk_size=256):
    text8_set = Text8Dataset(chunk_size=chunk_size)
    text8_loader = data.DataLoader(
        text8_set, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    return text8_loader

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
            data = torch.tensor(all_tokens, dtype=torch.uint8)
            torch.save(data, 'data/text8.pt')
            return data

    def __getitem__(self, index):
        # load textual data
        text = self.data[index]
        out = F.one_hot(text.long(), num_classes=self.num_tok).type(torch.float32)
        return rearrange(out, 'w k -> k w')

    def __len__(self):
        return self.data.shape[0]

def test_cifar10_dataset():
    batch_size = 4
    k = 10
    cifar10_loader = cifar10_dataset(batch_size, k)
    data_iter = iter(cifar10_loader)
    images, labels = next(data_iter)
    assert images.shape == (batch_size, 3, k, 32, 32)
    assert labels.shape == (batch_size,)

# TODO: check output of this...
if __name__ == "__main__":
    test_cifar10_dataset()
    # text8_test = Text8Dataset()
    # print(text8_test[0], text8_test[0].shape)
    # text8_test_loader = text8_dataset(32)
    # for batch in text8_test_loader:
    #     print(batch.shape)
    #     break
