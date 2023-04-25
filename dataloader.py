import os, sys
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

# convert to onehot encoding with k categories
class Onehot(object):
    def __init__(self, k=10):
        self.k = k

    def __call__(self, x):
        x *= self.k-1
        x = torch.round(x).squeeze().type(torch.int64) # remove channel dim
        x = one_hot(x, num_classes=self.k).type(torch.float32)
        return rearrange(x, 'h w k -> k h w')

# return mnist dataset
def mnist_dataset(batch_size, k, root='data/', num_workers=4, size=32):
    gray_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Grayscale(),
        transforms.Resize((size, size), antialias=True),
        Onehot(k)])
    mnist_set = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=gray_transform)
    mnist_loader = data.DataLoader(mnist_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return mnist_loader

# return dataset to main train and test framework
def celeba_dataset(batch_size, num_workers=4, size=64):
    celeba_transform =  transforms.Compose(
        [transforms.ToTensor(),
        transforms.CenterCrop((178, 178)), # square > rectangle
        transforms.Resize((size, size))]
    )

    # dataloader 
    celeba_set = CelebaDataset(img_path='/drive2/celeba/imgs', label_path='/drive2/celeba/attr.txt', transform=celeba_transform)
    celeba_loader = data.DataLoader(celeba_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return celeba_loader

# custom dataloader for celeba
class CelebaDataset(Dataset):
    def __init__(self, img_path, label_path, transform):
        self.img_path = img_path
        self.transform = transform

        # label map
        with open('/drive2/celeba/labels.txt', 'r') as f:
            labels = f.read()

        # load labels
        self.y = np.loadtxt(label_path, dtype=str)
        self.img_names = self.y[:,0]
        self.img_names = [name.replace('.jpg', '.png') for name in self.img_names]

    def __getitem__(self, index):
        # get images, then label
        img = Image.open(os.path.join(self.img_path, self.img_names[index]))
        label = self.y[index, 1:]

        # try using float16
        img = self.transform(img).type(torch.float32)
        label = torch.from_numpy(label.astype(np.float32))
        return img, label

    def __len__(self):
        return 202599

# custom dataloader for LM1b
class LM1BDataset(Dataset):
    def __init__(self):
        self.data = self.load_data()
        print('Data loaded.')

    def load_data(self):
        # check if data exists otherwise make GET request to data URL string
        # if data exists, load data
        if not os.path.exists('data/1-billion-word-language-modeling-benchmark-r13output'):
            if not os.path.exists('data'):
                os.makedirs('data')
            print('Downloading data...')
            url = 'http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz'
            r = requests.get(url, allow_redirects=True)
            open('data/1-billion-word-language-modeling-benchmark-r13output.tar.gz', 'wb').write(r.content)
            print('Download complete.')
            # unzip data
            import tarfile
            with tarfile.open('data/1-billion-word-language-modeling-benchmark-r13output.tar.gz', 'r') as tar_ref:
                tar_ref.extractall('data')
        # load data
        data = []
        for root, dirs, files in os.walk('data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'):
            for file in files:
                with open(os.path.join(root, file), 'r') as f:
                    data.extend(f.readlines())
        return data

    def __getitem__(self, index):
        # load textual data
        text = self.data[index]
        return text

    def __len__(self):
        return len(self.data)
    
# text8 dataset 
# data available at "http://mattmahoney.net/dc/text8.zip"
# based off of https://github.com/undercutspiky/Char_LM_PyTorch/blob/master/dataloader.py
class Text8Dataset(Dataset):
    def __init__(self, chunk_size=256):
        self.chunk_size = chunk_size
        self.data = self.load_data()
        print('Data loaded.')

    def load_data(self):
        # check if data exists otherwise make GET request to data URL string
        # if data exists, load data
        if not os.path.exists('data/text8'):
            if not os.path.exists('data'):
                os.makedirs('data')
            print('Downloading data...')
            url = 'http://mattmahoney.net/dc/text8.zip'
            r = requests.get(url, allow_redirects=True)
            open('data/text8.zip', 'wb').write(r.content)
            print('Download complete.')
            # unzip data
            import zipfile
            with zipfile.ZipFile('data/text8.zip', 'r') as zip_ref:
                zip_ref.extractall('data')
        # load data
        with open('data/text8', 'r') as f:
            data = f.read()
        # split data into chunks
        chunks = [data[i:i+self.chunk_size] for i in range(0, len(data), self.chunk_size)]
        return chunks

    def __getitem__(self, index):
        # load textual data
        text = self.data[index]
        return text

    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    text8_test = Text8Dataset()
    print(text8_test[0])