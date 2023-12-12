import torch
from einops import repeat
from torch.func import jacrev

# diffusion term for S process
def J(x):
    b, k = x.shape

    eye = torch.eye(k)
    I = repeat(eye, 'i j -> b i j', b=b).to(x.device)

    eq = repeat(x * (1 - x), 'b i ... -> b i j ...', j=k)
    neq = torch.einsum('b i ..., b j ... -> b i j ...', x, x)

    g1 = torch.einsum('b i j ..., b i j ... -> b i j ...', I, eq)
    g2 = torch.einsum('b i j ..., b i j ... -> b i j ...', 1-I, neq)

    return g1 - g2

def GG(x):
    return torch.einsum('b i j ..., b j k ... -> b i k ...', J(x), J(x))

def GG_grad(x, dt=1e-8):
    x1 = x - dt/2
    out_1 =  GG(x1)
    
    x2 = x + dt/2
    out_2 = GG(x2) 
    return (out_2 - out_1) / dt

# vectorized diffusion divergence
def diff_div(x):
    d = x.shape[1] + 1
    x_1 = x.sum(dim=1, keepdim=True)
    x_sq = x.square().sum(dim=1, keepdim=True)

    return x * ( (d + 2)*(x_sq - x) - 2*x_1 + 2 )

'''
this code checks that the analytical divergence of diffusion matricies is correct 
'''

torch.set_default_dtype(torch.float64)
if __name__ == '__main__':
    # distribution parameters
    batch_size = 1
    d = 4
    h = 32
    w = 32

    # get a point on the simplex
    x = torch.rand(batch_size, d, h, w)
    x /= x.sum(dim=1, keepdim=True)
    x = x[:, :-1]

    # get random pixels
    i_ = torch.randint(h, size=(1,)).item()
    j_ = torch.randint(w, size=(1,)).item()
    x_ = x[..., i_, j_]
    y = x_[0]

    # check that J^2 is correct
    print('numerical J^2')
    print(GG(x_))

    # first, look at diagonals
    i = 0
    Jdia = lambda y, i : y[i]**2 * ( (1-y[i])**2 - y[i]**2 + y.square().sum() )
    print(f'analytical J^2 at {i}-{i}')
    print(Jdia(y, i))

    # next, look at off diagonals
    i = 2; j = 1
    Joff = lambda y, i, j : -y[i]*y[j] * ( y[i]*(1-y[i]) + y[j]*(1-y[j]) + y[i]**2 + y[j]**2 - y.square().sum() )
    print(f'analytical J^2 at {i}-{j}')
    print(Joff(y, i, j))

    # compute numerical gradient
    print('\n\ncomputing gradients')

    #num_grad = GG_grad(x_)
    #num_div = num_grad.sum(dim=-1)
    #print('num grad')
    #print(num_grad)

    # get torch jacrev gradient 
    num_grad = jacrev(GG)(x_)
    num_grad = num_grad.squeeze()

    #print('torch grad')
    #for i in range(num_grad.shape[-1]):
        #print(num_grad[:,:,i])
    #print()

    # we want div: how the ith input changes the ith output
    print('div vectors')
    num_div = torch.zeros_like(num_grad[0])
    for i in range(num_grad.shape[-1]):
        num_div[:,i] = num_grad[i,:,i]
    print(num_div.sum(dim=1))
    print(num_div.sum(dim=0))
    print(num_div)

    i = 1
    dt = torch.zeros_like(y)
    dt[i] = 1e-6
    print(f'checking diagonal at: {i}-{i}')
    print(( Jdia(y+dt/2, i) - Jdia(y-dt/2, i) ) / dt)
    Hdia = lambda y, i : 2*y[i] * ( (1 - 3*y[i] + 2*y[i]**2) - y[i]**2 + y.square().sum() )
    print(Hdia(y, i))
    print()

    i = 1; j = 2
    dt = torch.zeros_like(y)
    dt[j] = 1e-6
    print(f'checking off-diagonal at: {i}-{j}')
    print(( Joff(y+dt/2, i, j) - Joff(y-dt/2, i, j) ) / dt)
    Hoff = lambda y, i, j : -y[i]**2*(1-y[i]) - y[i]*( y[j]*(2-3*y[j]) + y[i]**2 + y[j]**2 - y.square().sum() )

    print(Hoff(y, i, j))
    print()

    # testing
    print('testing')
    print(
        2*y* (y**2 -3*y + (1+y.square().sum()))
    )

    # get analytical gradient of log pdf
    ana_grad = diff_div(x)
    ana_grad = ana_grad[..., i_, j_]
    print('ana grad')
    print(ana_grad)
    quit()