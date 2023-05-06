import sys, os
import torch
import torch.distributions as dist
from einops import rearrange

from plot import save_vis, make_gif
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

        self.betas = torch.linspace(1e-4, 0.02, self.T)
        self.Q_bar = self.Q_bar(self.T).to(self.device)

        if args.dataset in ['mnist', 'cifar10']:
            self.data_type = 'image'
        elif args.dataset in ['text8']:
            self.data_type = 'text'

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

from dataloader import mnist_dataset, text8_dataset

# test noising process
if __name__ == '__main__':
    k = 10; T = 500; 
    data_type = 'text'
    methods = ['uniform', 'gaussian']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get data  
    if data_type == 'image':
        loader = mnist_dataset(8, k)
    elif data_type == 'text':
        k = 27
        loader = text8_dataset(8)

    (x, _) = next(iter(loader))
    print(x)
    sys.exit()

    for method in methods:
        x0 = x.clone().to(device)

        # get process
        process = CatProcess(k, T, method, data_type, device)

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

