import os, sys, yaml
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

from utils import get_model
from dataloaders import celeba_dataset
from plotting import save_vis, plot_loss

if __name__ == '__main__':
    # get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model from yaml
    yaml_path = 'configs/custom_vqgan.yaml'
    model = get_model(yaml_path).to(device)
    print(f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # get discriminator start
    with open(yaml_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    disc_start = config['model']['params']['lossconfig']['params']['disc_start']

    # load pretrained model
    #start = 66
    #model.load_state_dict(torch.load(f'results/model_{start}.pt'))

    # get loss function
    loss_fn = VQLPIPSWithDiscriminator(disc_start=disc_start).to(device)
    print(f'parameters: {sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)}')

    # optimizers
    opt_0 = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_1 = torch.optim.Adam(loss_fn.parameters(), lr=1e-10)

    # make dataloader
    celeba_loader = celeba_dataset(batch_size=128, num_workers=4, size=64)

    # train
    step = disc_start
    loss = []
    for epoch in range(500):
        sub_loss = []
        for i, (x, y) in enumerate(tqdm(celeba_loader)):
            opt_0.zero_grad()
            opt_1.zero_grad()

            x = x.to(device)
            x_out, book_loss, z, codes = model(x)

            loss_0, _ = loss_fn(book_loss, x, x_out, 0, step, last_layer=x_out)
            loss_0.backward()
            opt_0.step()
            
            sub_loss.append(loss_0.item())

            #loss_1, _ = loss_fn(book_loss, x, x_out, 1, step, last_layer=x_out)
            #loss_1.backward()
            #opt_1.step()
        
        loss.append(np.mean(sub_loss))

        # save information    
        save_vis(x, x_out, epoch, path='results')
        plot_loss(loss, path='results')
        torch.save(model.state_dict(), f'./results/model_{epoch}.pt')
        #step -= 1