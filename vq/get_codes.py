import os, sys
import argparse
import yaml
import torch
from tqdm import tqdm

from taming.models.vqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

from utils import get_model
from dataloaders import celeba_dataset

# arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='results/second/model_121.pt')
    parser.add_argument('--batch_size', type=int, default=1028)

    args = parser.parse_args()
    return args

# get all codes from celeba dataset
if __name__ == '__main__':
    args = get_args()
    assert args.model_path is not None

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load yaml file
    yaml_path = 'configs/custom_vqgan.yaml'
    model = get_model(yaml_path)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # dataloder
    celeba_loader = celeba_dataset(args.batch_size)

    # get codes
    all_codes = None
    with torch.no_grad():
        for batch in tqdm(celeba_loader):
            img, label = batch
            img = img.to(device)

            # forward pass
            out, enc_loss, z, codes = model(img)

            # save codes for dir diff
            if all_codes is None: all_codes = codes[None, ...]
            else: all_codes = torch.cat((all_codes, codes[None, ...]), dim=0)

    # save codes
    torch.save(all_codes, 'data/codes.pt')