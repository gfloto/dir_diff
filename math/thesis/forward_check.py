import torch
from einops import repeat

# from https://arxiv.org/pdf/2011.13456.pdf: page 16
def beta(t, beta_min=0.1, beta_max=20):
    #return torch.tensor(1.)
    return (beta_max - beta_min) * t + beta_min

# map from reals to simplex
def sig(z):
    return z.exp() / (1 + z.exp().sum(dim=1, keepdim=True))

# map from simplex to reals
def sig_inv(x, eps=1e-8):
    out = x.log() - (1 - x.sum(dim=1, keepdim=True)).log()
    return out

# drift term for S process
def f(x, t):
    m = x * (1 - 2*x)
    g = m - m.sum(dim=1, keepdim=True) * x
    f = torch.einsum('b i j ..., b j ... -> b i ...', J(x, t), sig_inv(x))

    return -0.5*beta(t)*f + 0.5*g

# diffusion term for S process
def J(x, t):
    b, k, w, h = x.shape

    eye = torch.eye(k)
    I = repeat(eye, 'i j -> b i j w h', b=b, w=w, h=h).to(x.device)

    eq = repeat(x * (1 - x), 'b i ... -> b i j ...', j=k)
    neq = torch.einsum('b i ..., b j ... -> b i j ...', x, x)

    g1 = torch.einsum('b i j ..., b i j ... -> b i j ...', I, eq)
    g2 = torch.einsum('b i j ..., b i j ... -> b i j ...', 1-I, neq)

    return g1 - g2

# forward process differential for S
def update_x(x, t, dt, eps):
    dB = eps * dt.sqrt()
    JdB = torch.einsum('b i j ..., b j ... -> b i ...', J(x, t), dB)

    #x = x + f(x, t)*dt + JdB
    x = x + f(x, t)*dt + beta(t).sqrt() * JdB
    return x

# forward process differential for R
def update_z(z, t, dt, eps):
    #z += -0.5*beta(t)*z*dt + eps * dt.sqrt()
    z += -0.5*beta(t)*z*dt + beta(t).sqrt() * eps * dt.sqrt()
    return z 

'''
this code checks that the applications of ito's lemma is correct

a diffusion process is ran in R^d and in S^d
which are checked to be equivalent
'''

#torch.set_default_dtype(torch.float64)
if __name__ == '__main__':
    # distribution parameters
    batch_size = 128
    d = 4
    h = 1
    w = 1

    # get batch of points in R and S
    z = torch.randn(1, d-1, h, w)
    z = z.repeat(batch_size, 1, 1, 1)
    x = sig(z) 

    # run diffusion process in R and S 
    T = 1000; dt = torch.tensor(1/T)
    t_ = torch.arange(T) * dt + dt

    for i in range(T):
        t = t_[i]
        eps = torch.randn_like(z)

        # run diffusion process in R
        z = update_z(z, t, dt, eps)
        x = update_x(x, t, dt, eps)
        x_inv = sig_inv(x)

        # check that the diffusion process in R and S are equivalent
        r_mean = z.mean(dim=(0,2,3)); r_std = z.std(dim=(0,2,3))
        s_mean = x_inv.mean(dim=(0,2,3)); s_std = x_inv.std(dim=(0,2,3))

        if i % 100 == 0:
            print(r_mean, s_mean)
            print(r_std, s_std)
            print(x.sum(dim=1)[:5, 0, 0])
            print()