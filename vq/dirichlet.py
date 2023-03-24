import os, sys
import torch
import argparse
from dataloaders import CodebookDataset

# get args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get dataloader
    loader = CodebookDataset('data/codes.pt', args)

    for i in range(loader.n_batch):
        x = loader.get_batch(i)
        print(x.shape)
        break
