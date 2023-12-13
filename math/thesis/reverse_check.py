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
    f = torch.einsum('b i j ..., b j ... -> b i ...', J(x), sig_inv(x))

    return -0.5*beta(t)* (g - f)

# diffusion term for S process
def J(x):
    b, k, w, h = x.shape

    eye = torch.eye(k)
    I = repeat(eye, 'i j -> b i j w h', b=b, w=w, h=h).to(x.device)

    eq = repeat(x * (1 - x), 'b i ... -> b i j ...', j=k)
    neq = torch.einsum('b i ..., b j ... -> b i j ...', x, x)

    g1 = torch.einsum('b i j ..., b i j ... -> b i j ...', I, eq)
    g2 = torch.einsum('b i j ..., b i j ... -> b i j ...', 1-I, neq)

    return g1 - g2

def G(x, t):
    return beta(t).sqrt() * J(x)

def GG(x, t):
    return torch.einsum('b i j ..., b j k ... -> b i k ...', G(x, t), G(x, t))

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

# divergence term for backwards S process
def div_GG(x, t):
    d = x.shape[1] + 1
    x_1 = x.sum(dim=1, keepdim=True)
    x_sq = x.square().sum(dim=1, keepdim=True)

    return beta(t) * x * ( (d + 2)*(x_sq - x) - 2*x_1 + 2 )

# helper function
def gamma(x, mu):
    if x.log().isnan().any(): print('hit 1')
    if (1 - x.sum(dim=1, keepdim=True)).log().isnan().any(): print('hit 2')

    return x.log() - (1 - x.sum(dim=1, keepdim=True)).log() - mu

# score or grad log pdf
def logistic_score(x, mu, v):
    g = gamma(x, mu)

    xd = 1 - x.sum(dim=1, keepdim=True)

    a1 = (x - xd) / (x * xd)
    a2 = g / x
    a3 = g.sum(dim=1, keepdim=True) / xd

    return a1 - (a2 + a3) / v

# p(z_t | z_0) in R
def p_zt_z0(z0, t, beta_min=0.1, beta_max=20):
    b_diff = beta_max - beta_min
    mu = (-0.25 * t**2 * b_diff - 0.5 * t * beta_min).exp() * z0
    v = 1 - (-0.5 * t**2 * b_diff - t * beta_min).exp()

    zt = mu + v.sqrt() * torch.randn_like(z0)
    return  zt, mu, v

# p(x_t | x_0) in S
# from https://arxiv.org/pdf/2011.13456.pdf: page 16
def p_xt_x0(x0, t, beta_min=0.1, beta_max=20):
    z0 = sig_inv(x0)

    b_diff = beta_max - beta_min
    mu = (-0.25 * t**2 * b_diff - 0.5 * t * beta_min).exp() * z0
    v = 1 - (-0.5 * t**2 * b_diff - t * beta_min).exp()

    zt = mu + v.sqrt() * torch.randn_like(z0)
    xt = sig(zt)
    return xt, mu, v

# forward process differential for S
def update_x(x, t, score, dt, eps):
    GdB = torch.einsum('b i j ..., b j ... -> b i ...', G(x, t), eps * dt.sqrt())
    GG_score = torch.einsum('b i j ..., b j ... -> b i ...', GG(x, t), score)

    x -= ( f(x, t) - 0.5*div_GG(x, t) - 0.5*GG_score )*dt + GdB
    return x

# forward process differential for R
def update_z(z, t, score, dt, eps):
    a = (-0.5*beta(t)*z - beta(t)*score) * dt
    b = beta(t).sqrt()*eps*dt.sqrt()
    z -= a + b
    return z 

'''
this code checks that the applications of ito's lemma is correct
(reverse direction only)

a diffusion process is ran in R^d and in S^d
which are checked to be equivalent
'''

torch.set_default_dtype(torch.float64)
if __name__ == '__main__':
    device = 'cuda'

    # distribution parameters
    batch_size = 128
    d = 11
    h = 32
    w = 32

    # get initial data point
    v1 = 0.1; v0 = v1 / (d-1)
    x_init = torch.zeros(1, d-1, 1, 1) + v0
    idx = torch.randint(d-1, size=(1,)).item()
    if idx < d-1: x_init[:, idx, ...] = 1-v1
    print(f'using {v0:.5f} and {1-v1:.5f} for initial data point')

    # get batch of points in R and S
    x0 = x_init.repeat(batch_size, 1, h, w).to('cuda')
    z0 = sig_inv(x0) 

    # get initial points in R and S
    xT, _, _ = p_xt_x0(x0, torch.tensor(1.))
    zT, _, _ = p_zt_z0(z0, torch.tensor(1.))
    print(f'initial z0 mean: {z0.mean(dim=(0,2,3)).cpu()}')
    print()

    # run diffusion process in R and S 
    T = 1000; dt = torch.tensor(1/T)
    t_ = torch.arange(T) * dt + dt

    x = xT.clone(); z = zT.clone()
    for i in range(T-1, -1, -1):
        t = t_[i]
        eps = torch.randn_like(z)

        # run diffusion process in R
        zt, z_mu, z_v = p_zt_z0(z0, t)
        z_score = -(z - z_mu) / z_v
        z = update_z(z, t, z_score, dt, eps)

        # run diffusion process in S
        xt, x_mu, x_v = p_xt_x0(x0, t)
        x_score = logistic_score(x, x_mu, x_v)
        x = update_x(x, t, x_score, dt, eps)

        # clip x
        c = 1e-8
        x = torch.clamp(x, c, 1 - c)

        # check that the diffusion process in R and S are equivalent
        z_mean = z.mean(dim=(0,2,3)); z_std = z.std(dim=(0,2,3))
        zt_mean = zt.mean(dim=(0,2,3)); zt_std = zt.std(dim=(0,2,3))

        x_mean = x.mean(dim=(0,2,3)); x_std = x.std(dim=(0,2,3))
        x_inv_mean = sig_inv(x).mean(dim=(0,2,3)); x_inv_std = sig_inv(x).std(dim=(0,2,3))
        
        xt_mean = xt.mean(dim=(0,2,3)); xt_std = xt.std(dim=(0,2,3))
        xt_inv_mean = sig_inv(xt).mean(dim=(0,2,3)); xt_inv_std = sig_inv(xt).std(dim=(0,2,3))

        if i % 100 == 0 or i == T-1:
            print(z_mean.cpu())
            print(zt_mean.cpu())
            print(xt_inv_mean.cpu())
            #print(x_inv_mean.cpu())
            #print(x_mean.cpu())

            #print( (z_mean - zt_mean).abs().cpu(), (z_std - zt_std).abs().cpu() )
            print()
