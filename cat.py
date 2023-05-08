import sys, os
import torch
import torch.distributions as dist
import numpy as np
from einops import rearrange

from utils import cat2onehot

# useful sampling function
def sample(q, k):
    q = rearrange(q, 'b k ... -> b ... k')
    out = dist.Categorical(q).sample()
    return cat2onehot(out, k=k, shape=q.shape)

# shortname for using einsum
def mm(q, x):
    return torch.einsum('ik, bk... -> bi...', q, x)

# TODO: where are the embeddings coming from?
def compute_knn_adjacency_matrix(embeddings, k):
    # Compute the k-nearest neighbors of each word in the embedding space
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    # Construct the k-nearest neighbor adjacency matrix
    n_words = embeddings.shape[0]
    G = np.zeros((n_words, n_words))
    for i in range(n_words):
        G[i, indices[i]] = 1

    return G

'''
process for default categorical diffusion
see: https://arxiv.org/pdf/2107.03006.pdf
'''

class CatProcess:
    def __init__(self, args):
        self.k = args.k
        self.T = args.T
        self.method = args.q_method
        self.device = args.device

        # useful attributes
        if args.dataset in ['mnist', 'cifar10']:
            self.data_type = 'image'

        elif args.dataset in ['text8']:
            self.data_type = 'text'
            self.char2idx = {char: idx for idx, char in enumerate(
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' '])}
            self.idx2char = {idx: char for char, idx in self.char2idx.items()}

        self.sched_method = args.sched_method
        self.betas = self.create_betas(self.T, sched_method=self.sched_method)
        self.Q_bar = self.Q_bar(self.T).to(self.device)

    # create betas for the diffusion process
    # based on the noise schedule method
    def create_betas(self, T, sched_method='linear'):
        if sched_method == 'linear':
            betas = torch.linspace(1e-4, 0.02, T)
        elif sched_method == 'cosine':
            s = 0.008
            f = lambda t, s: np.cos(np.pi/2 * (t/T + s)/(1+s))
            betas = [1 - f(t+1, s)/f(0, s) for t in range(T)]
            betas = torch.tensor(betas)
        elif sched_method == 'mutual_info':
            betas = [1/(T-t+1) for t in range(T)]
            betas = torch.tensor(betas)
        else: 
            raise ValueError('Invalid schedule method')
        return betas
    
    # get t, rescale to be in proper interval
    def t(self):
        t = torch.randint(self.T, (1,))
        tu = t / self.T
        return t.item(), tu.to(self.device)

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

        elif self.method == 'absorbing':
            # if the data is an image m is set to (128, 128, 128) at index K//2
            # if the data is text m is set to [MASK] at index K-1
            m = self.k // 2 if self.data_type == 'image' else self.k - 1
            beta_t = self.betas[t]
            Qt = (1 - beta_t) * torch.eye(self.k)
            Qt[:, m] += beta_t

        elif self.method == 'gaussian':
            beta_t = self.betas[t]
            Qt = torch.zeros(self.k, self.k)
            beta_t = torch.tensor(beta_t)
            normalization = torch.sum(torch.exp(-4 * (torch.arange(-(self.k - 1), self.k) ** 2) / ((self.k - 1) ** 2 * beta_t)))
            i, j = torch.meshgrid(torch.arange(self.k), torch.arange(self.k))
            Qt = torch.exp(-4 * (i - j) ** 2 / ((self.k - 1) ** 2 * beta_t)) / normalization
            Qt[range(self.k), range(self.k)] = 0
            Qt[range(self.k), range(self.k)] = 1 - Qt.sum(dim=1)

        elif self.method == 'knn':
            # Compute the pairwise distances between all words in the embedding space
            distances = torch.cdist(embeddings, embeddings)

            # Find the k-nearest neighbors of each word
            # Exclude the first nearest neighbor, which is the word itself
            knn = distances.topk(k=k+1, dim=1, largest=False).indices[:, 1:]

            # Construct the k-nearest neighbor adjacency matrix
            n_words = embeddings.shape[0]
            G = torch.zeros((n_words, n_words))
            G.scatter_(1, knn, 1)

            # Symmetrize the adjacency matrix
            A = (G + G.T) / (2 * k)

            # Construct the rate matrix R by modifying A directly
            A[range(n_words), range(n_words)] = -A.sum(dim=1)

            # Compute the transition matrix using a matrix exponential
            alpha_t = alphas[t]
            Qt = torch.matrix_exp(alpha_t * A) 

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
    def Q_rev(self, x0, xt, t):
        num =  mm(self.Q(t).T, xt) * mm(self.Q_bar[t-1], x0)
        denom = torch.einsum('bk..., bk... -> b...', xt, mm(self.Q_bar[t], x0))
        denom = torch.stack(self.k*[denom], dim=1)
        out = num / denom

        return out 

    # one-hot text to string
    def decode_text(self, encoded_text):
        # Convert one-hot encoding to indices
        indices = torch.argmax(encoded_text, dim=1)

        # Convert indices to characters using the idx2char dictionary
        decoded_text = [''.join([self.idx2char[idx.item()]
                                for idx in example]) for example in indices]
        return decoded_text 

from tqdm import tqdm
from args import get_args
from plot import save_vis, make_gif
from dataloader import text8_dataset, mnist_dataset, cifar10_dataset

# test noising process
if __name__ == '__main__':
    chars = 50
    # get device, data and process
    args = get_args()
    if args.dataset == 'text8':
        loader = text8_dataset(args.batch_size)
    elif args.dataset == 'mnist':
        loader = mnist_dataset(args.batch_size, args.k)
    elif args.dataset == 'cifar10':
        loader = cifar10_dataset(args.batch_size, args.k)
    process = CatProcess(args)

    # get x0
    x0 = next(iter(loader))
    if isinstance(x0, tuple) or isinstance(x0, list):
        x0 = x0[0] 
    x0 = x0.to(args.device)

    # print initial text
    if process.data_type == 'text':
        print(f't: {0:.3f}, text: {process.decode_text(x0)[0][:chars]}')

    # test forward process
    os.makedirs('imgs', exist_ok=True)
    for t in range(args.T):
        qbar = process.Q_bar[t]

        # apply qbar to x0
        xt = process.xt(x0, t)

        # save image
        if process.data_type == 'image':
            save_vis(xt, f'imgs/{t}.png', k=args.k)
        else:
            print(f't: {t/args.T:.3f}, text: {process.decode_text(xt)[0][:chars]}')

    # make gif of forward process
    if process.data_type == 'image':
        make_gif('imgs', f'results/forward_cat_{args.dataset}_{args.q_method}.gif', args.T)

