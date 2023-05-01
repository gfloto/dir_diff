import sys, os
import torch
import torch.distributions as dist
from einops import rearrange

from plot import save_vis, make_gif
from utils import cat2onehot

# useful sampling function
def sample(q, k):
    q = rearrange(q, 'b k h w -> b h w k')
    out = dist.Categorical(q).sample()
    return cat2onehot(out, k=k)

# shortname for using einsum
def mm(q, x):
    return torch.einsum('ik, bkwh -> biwh', q, x)

'''
process for default categorical diffusion
see: https://arxiv.org/pdf/2107.03006.pdf
'''

class CatProcess:
    def __init__(self, k, T, betas, method, device):
        self.k = k
        self.T = T
        self.betas = betas
        self.method = method
        self.data_type = 'image'
        self.device = device

        self.Q_bar = self.Q_bar(T)

    # forward process
    def xt(self, x0, t):
        # sample from q(xt | x0)
        p = mm(self.Q_bar[t], x0)
        xt = sample(p, self.k) 
        return xt

    # compute Qt transition matrix 
    def Q(self, t):
        method = self.method
        if method == 'uniform':
            b = self.betas[t]; k = self.k
            Qt = (1-b) * torch.eye(k) + b*torch.ones(k,k) / k

        elif method == 'absorbing':
            # if the data is an image m is set to (128, 128, 128) at index K//2
            # if the data is text m is set to [MASK] at index K-1
            m = self.k // 2 if self.data_type == 'image' else self.k - 1
            beta_t = self.betas[t]
            Qt = (1 - beta_t) * torch.eye(self.k)
            Qt[:, m] += beta_t

        elif method == 'gaussian':
            beta_t = self.betas[t]
            Qt = torch.zeros(self.k, self.k)
            beta_t = torch.tensor(beta_t)
            normalization = torch.sum(torch.exp(-4 * (torch.arange(-(self.k - 1), self.k) ** 2) / ((self.k - 1) ** 2 * beta_t)))
            i, j = torch.meshgrid(torch.arange(self.k), torch.arange(self.k))
            Qt = torch.exp(-4 * (i - j) ** 2 / ((self.k - 1) ** 2 * beta_t)) / normalization
            Qt[range(self.k), range(self.k)] = 0
            Qt[range(self.k), range(self.k)] = 1 - Qt.sum(dim=1)

        return Qt.to(self.device)

    # Q_bar is q(x_t | x_0) (which is just Q_1 @ Q_2 @ ...)
    def Q_bar(self, t):
        Qt_bar = torch.zeros((t, self.k, self.k)).to(self.device)
        Qt_bar[0] = self.Q(0)
        for i in range(1,t):
            Qt = self.Q(i)
            Qt_bar[i] = Qt_bar[i-1] @ Qt
        return Qt_bar

    # fill in later
    def q_rev(self, x0, xt, t):
        num =  mm(self.Q(t).T, xt) * mm(self.Q_bar[t-1], x0)
        denom = torch.einsum('bkhw, bkhw -> bhw', xt, mm(self.Q_bar[t], x0))
        denom = torch.stack(self.k*[denom], dim=1)
        out = num / denom
        if not out.sum(dim=1).allclose(torch.ones_like(out.sum(dim=1))):
            a = out.sum(dim=1)
            print(t)
            print(a[torch.where(a != 1)])
        return out 

from dataloader import mnist_dataset

# test noising process
if __name__ == '__main__':
    k = 10; T = 500; 
    methods = ['uniform', 'absorbing', 'gaussian', 'gaussian_vectorized']
    betas = torch.linspace(1e-4, 0.02, T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data  
    loader = mnist_dataset(8, k)
    (x, _) = next(iter(loader))

    for method in methods:
        x0 = x.clone().to(device)

        # get process
        process = CatProcess(k, T, betas, method, device)

        # test forward process
        r = 10
        print(f'running {method} forward process...')
        os.makedirs('imgs', exist_ok=True)
        for t in range(T):
            qbar = process.Q_bar[t]

            # apply qbar to x0
            xt = process.xt(x0, t)

            # save image
            if t % r == 0:
                save_vis(xt, f'imgs/{int(t/r)}.png', k=k, n=8)

        # make gif of forward process
        make_gif('imgs', f'results/cat_{method}.gif', T//r)

