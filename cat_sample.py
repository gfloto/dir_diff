import sys, os
import torch
import numpy as np
from tqdm import tqdm
from model import Unet
from cat import sample
from utils import cat2onehot, onehot2cat
from plot import save_vis, make_gif


@torch.no_grad()
def cat_sample(model, T=1000, batch=8, save_path='sample.gif'):
    d = 10
    if save_path is not None:
        os.makedirs('imgs', exist_ok=True)

    # get initial distribution
    x = torch.rand((batch, 2, 32, 32)).to('cuda')
    t_ = np.linspace(0, 1, T)[::-1]

    # sample from model
    for i in tqdm(range(T)):
        t = torch.tensor([t_[i]]).type(torch.float32).to('cuda')
        qrev = model(x, t)
        x = sample(qrev.exp()) # perhaps just sample a onehot?

        # save image
        if i % d == 0:
            save_vis(x, f'imgs/{int(i/d)}.png', k=2)

        # convert to one hot
        x = cat2onehot(onehot2cat(x, 2).long(), 2)

    # make gif
    make_gif(save_path)


# sorts files alpha numerically (natural computer display)
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


if __name__ == '__main__':
    # get newest model from f'results/model{num}.pt'
    names = [f for f in os.listdir('results/') if 'model_cat_' in f]
    names = sorted(names, key=natural_key)
    model = Unet(dim=64, channels=2).to('cuda')
    model.load_state_dict(torch.load(f'results/{names[-1]}'))
    model.eval() 

    # print model name
    print(f'Using model: {names[-1]}')

    # sample from model
    cat_sample(model, T=1000, batch=8, save_path='cat_sample.gif') 
