import os, sys
import argparse
import yaml
import torch
from tqdm import tqdm

from basic import celeba_dataset
from taming.models.vqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

# arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='results/model_109.pt')
    parser.add_argument('--batch_size', type=int, default=1028)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    assert args.model_path is not None

    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load yaml file
    yaml_path = 'configs/custom_vqgan.yaml'
    with open(yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    ddconfig = config['model']['params']['ddconfig']
    lossconfig = config['model']['params']['lossconfig']
    n_embed = config['model']['params']['n_embed']
    embed_dim = config['model']['params']['embed_dim']
    disc_start = config['model']['params']['lossconfig']['params']['disc_start']
    print(f'embed dim: {embed_dim}\t num embed: {n_embed}')

    # make model    
    model = VQModel(ddconfig, lossconfig, n_embed, embed_dim).to(device)
    model.load_state_dict(torch.load(args.model_path))

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
            if all_codes is None: all_codes = codes
            else: all_codes = torch.cat((all_codes, codes), dim=0)

    # save codes
    torch.save(all_codes, 'data/codes.pt')