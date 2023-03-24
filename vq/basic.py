import os, sys, yaml
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch.utils.data as data
from torchvision import transforms
from torch.utils.data import Dataset

from taming.models.vqgan import VQModel
from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

from plotting import save_vis, plot_loss

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

if __name__ == '__main__':
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

    # make model, feel shame for your small neural networks    
    model = VQModel(ddconfig, lossconfig, n_embed, embed_dim).to(device)
    print(f'parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # load pretrained model
    start = 66
    model.load_state_dict(torch.load(f'results/model_{start}.pt'))

    # get loss function
    loss_fn = VQLPIPSWithDiscriminator(disc_start=disc_start).to(device)
    print(f'parameters: {sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)}')

    # optimizers
    opt_0 = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_1 = torch.optim.Adam(loss_fn.parameters(), lr=1e-10)

    # make dataloader
    celeba_loader = celeba_dataset(batch_size=64, num_workers=4, size=64)

    # train
    step = disc_start
    loss = []
    for epoch in range(start+1, 500):
        sub_loss = []
        for i, (x, y) in enumerate(tqdm(celeba_loader)):
            opt_0.zero_grad()
            opt_1.zero_grad()

            x = x.to(device)
            x_out, book_loss, lat = model(x)

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