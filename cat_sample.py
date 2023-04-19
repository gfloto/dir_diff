import sys, os
import torch
from tqdm import tqdm

from model import Unet
from cat import sample
from utils import cat2onehot
from plot import save_vis, make_gif

@torch.no_grad()
def sample(model, T=1000, batch=8, save_path='sample.gif'):
    d = 10
    if save_path is not None:
        os.makedirs('imgs', exist_ok=True)

    # get initial distribution
    x = torch.randn(batch, 2, 32, 32).to('cuda')

    # sample from model
    for t in tqdm(range(T)):
        qrev = model(x, torch.tensor([t/T]).to('cuda'))
        x = qrev.sample() # perhaps just sample a onehot?

        # save image
        if t % d == 0:
            save_vis(x, f'frames/{int(T/d)}.png')

        # convert to one hot
        x = cat2onehot(x)

    # make gif
    make_gif(save_path)

# sorts files alpha numerically (natural computer display)
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

if __name__ == '__main__':
    # get newest model from f'results/model{num}.pt'
    names = [f for f in os.listdir('results/') if 'model' in f]
    names = sorted(names, key=natural_key)
    model = Unet(dim=64, channels=1).to('cuda')
    model.load_state_dict(torch.load(f'results/{names[-1]}'))
    model.eval() 

    # print model name
    print(f'Using model: {names[-1]}')

    # sample from model
    sample(model, T=1000, batch=8, save_path='cat_sample.gif') 