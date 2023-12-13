import torch
from einops import repeat

# from https://arxiv.org/pdf/2011.13456.pdf: page 16
def beta(t, beta_min=0.1, beta_max=20):
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

    return 0.5*beta(t)* (g - f)

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

def G(x, t):
    return beta(t).sqrt() * J(x, t)

# forward process differential for S
def update_x(x, t, dt, eps):
    dB = eps * dt.sqrt()
    GdB = torch.einsum('b i j ..., b j ... -> b i ...', G(x, t), dB)

    x += f(x, t)*dt + GdB
    return x

# forward process differential for R
def update_z(z, t, dt, eps):
    z += -0.5*beta(t)*z*dt + beta(t).sqrt() * eps * dt.sqrt()
    return z 

# p(z_t | z_0) in R
def p_zt_z0(z0, t, beta_min=0.1, beta_max=20):
    b_diff = beta_max - beta_min
    mu = (-0.25 * t**2 * b_diff - 0.5 * t * beta_min).exp() * z0
    v = 1 - (-0.5 * t**2 * b_diff - t * beta_min).exp()

    zt = mu + v.sqrt() * torch.randn_like(z0)
    return  zt

# p(x_t | x_0) in S
# from https://arxiv.org/pdf/2011.13456.pdf: page 16
def p_xt_x0(x0, t, beta_min=0.1, beta_max=20):
    z0 = sig_inv(x0)

    b_diff = beta_max - beta_min
    mu = (-0.25 * t**2 * b_diff - 0.5 * t * beta_min).exp() * z0
    v = 1 - (-0.5 * t**2 * b_diff - t * beta_min).exp()

    zt = mu + v.sqrt() * torch.randn_like(z0)
    xt = sig(zt)
    return xt

'''
this code checks that the applications of ito's lemma is correct

a diffusion process is ran in R^d and in S^d
which are checked to be equivalent
'''

#torch.set_default_dtype(torch.float64)
if __name__ == '__main__':
    device = 'cuda'

    # distribution parameters
    batch_size = 128
    d = 4
    h = 32
    w = 32

    # get initial data point
    v1 = 1e-1; v0 = v1 / (d-1)
    x_init = torch.zeros(1, d-1, 1, 1) + v0
    idx = torch.randint(d-1, size=(1,)).item()
    if idx < d-1: x_init[:, idx, ...] = 1-v1
    print(f'using {v0:.5f} and {1-v1:.5f} for initial data point')

    # get batch of points in R and S
    x0 = x_init.repeat(batch_size, 1, h, w).to('cuda')
    z0 = sig_inv(x0) 

    # run diffusion process in R and S 
    T = 1000; dt = torch.tensor(1/T)
    t_ = torch.arange(T) * dt + dt

    x = x0.clone(); z = z0.clone()
    for i in range(T):
        t = t_[i]
        eps = torch.randn_like(z)

        # run diffusion process in R
        z = update_z(z, t, dt, eps)
        zt = p_zt_z0(z0, t)

        x = update_x(x, t, dt, eps)
        xt = p_xt_x0(x0, t)

        x_inv = sig_inv(x)
        xt_inv = sig_inv(xt)

        # check that the diffusion process in R and S are equivalent
        z_mean = z.mean(dim=(0,2,3)); z_std = z.std(dim=(0,2,3))
        zt_mean = zt.mean(dim=(0,2,3)); zt_std = zt.std(dim=(0,2,3))
        
        x_mean = x.mean(dim=(0,2,3)); x_std = x.std(dim=(0,2,3))
        xt_mean = xt.mean(dim=(0,2,3)); xt_std = xt.std(dim=(0,2,3))

        x_inv_mean = x_inv.mean(dim=(0,2,3)); x_inv_std = x_inv.std(dim=(0,2,3))
        xt_inv_mean = xt_inv.mean(dim=(0,2,3)); xt_inv_std = xt_inv.std(dim=(0,2,3))

        if i % 100 == 0:
            #print(x_mean.cpu())
            #print(xt_mean.cpu())

            print( (z_mean - zt_mean).abs().cpu(), (z_std - zt_std).abs().cpu() )
            print( (x_mean - xt_mean).abs().cpu(), (x_std - xt_std).abs().cpu() )
            print( (z_mean - xt_inv_mean).abs().cpu(), (z_std - xt_inv_std).cpu() )
            print()
