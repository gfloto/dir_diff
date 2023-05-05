import sys, os
import torch
import numpy as np
from einops import repeat
from tqdm import tqdm
import matplotlib.pyplot as plt

from args import get_args
from plot import make_gif, scatter_plot, vector_plot
from process import sig, Process
from dataloader import mnist_dataset

if __name__ == '__main__':
    T = 100; N = 64
    args = get_args()
    args.k=4
    process = Process(args)

    # make data to show each of k=3 processes
    x0 = torch.zeros(N, args.k, 32, 32)
    x1 = torch.zeros(N, args.k, 32, 32)
    x2 = torch.zeros(N, args.k, 32, 32)
    x0[:,0] = 1
    x1[:,1] = 1
    x2[:,2] = 1
    x = [x0, x1, x2]

    os.makedirs('imgs', exist_ok=True)
    t = np.linspace(args.t_min, args.t_max, T)
    for i in tqdm(range(T)):
        x_out, g_out = [], []
        for j in range(len(x)):
            xt, mu, var = process.xt(x[j].clone(), t[i])
            g2_score = process.g2_score(xt, mu, var)

            # squeeze
            xt = xt[:,:,0,0]
            g2_score = g2_score[:,:,0,0]
            g2_score = 0.01* g2_score # for plotting clarity

            # save
            x_out.append(xt)
            q = torch.stack([xt, g2_score], dim=0)
            g_out.append(q)

        # plot together to compare
        #scatter_plot(x_out, i, f'imgs/{i}.png')
        vector_plot(g_out, i, f'imgs/{i}.png')

    make_gif('imgs', 'g2score_v_check.gif', T)

