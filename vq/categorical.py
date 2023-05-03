import torch
import torch.nn as nn
import torch.distributions as dist
from labml_nn.diffusion.ddpm.utils import gather

# import the unet model 
from ..model import Unet

class CategoricalDiffusionGrayscale(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 num_hidden_channels, K, T, beta_values
                 use_gumbel=True, tau=1.0):
        super(CategoricalDiffusionGrayscale, self).__init__()

        # U-Net for predicting noise schedule
        # Subject to change 
        self.unet = Unet(dim=num_hidden_channels, channels=K)
        
        # we don't need this
        self.tau = tau  # Temperature parameter, adjust as needed
        
        self.K = K
        self.T = T # re-label this T
        self.beta_values = beta_values
        self.ones = torch.ones(K, K)
        self.Q_t = self.compute_Q_t(T)
        self.Q_bar = self.make_Q_bar(T)
                
    # Compute the Q_t matrix for a given timestep t
    # using the Uniform distribution defined in Appendix A.2.1
    def compute_Q_t(self, t):
        beta_t = self.beta_values[t]
        Q_t = (1 - beta_t) * torch.eye(self.K) + beta_t * self.ones / self.K
        return Q_t
    
    # Q_bar is q(x_t | x_0) (which is just Q_0 @ Q_1 @ ...)
    # DONE: move this to a tensor of self.Q_bar: shape [T, k, k]
    def make_Q_bar(self, t):
        result_tensor = torch.empty((t, self.K, self.K))
        bar_Q_t = torch.eye(self.K)
        for i in range(t):
            bar_Q_t = bar_Q_t @ self.compute_Q_t(i)
            result_tensor[i] = bar_Q_t
        return result_tensor

    # DONE: q(x_{t} | x_{t-1}, x_0) =q(x_{t} | x_{t-1}) = Q_t @ x_{t-1}
    # see equation 3
    def make_Q_rev(self, x0, t):
        x_t_m1 = self.forward_diffusion(x0, self.Q_bar[t-2]) # self.Q_bar is zero-indexed so x_{t-1} is at index t-2
        q_rev = self.Q_t @ x_t_m1
        return q_rev

    
    # NOTE: each of the things above can be used with some x s.t. Q@x = a probability distribution p
    # we can then sample from this p using a standard pytorch categorical
    # thus, Q_t, and Q_bar_t should be stored in the class and never re-computed

    # Add noise to the image x0 using the forward diffusion process
    def forward_diffusion(self, x0, Q_bar):
        p = torch.matmul(x0, Q_bar) # ie. probability that you're in state xt given x0
        # Sample directly from the categorical distribution
        x = dist.Categorical(p).sample()
        return x

    
    # q(x_{t-1}|x_t, x_0)
    def create_q_tm_1(self, x0, t):
        q_rev = self.make_Q_rev(x0, t)
        return (q_rev @ self.Q_bar[t-2]) / self.Q_bar[t-1] # self.Q_bar is zero-indexed so x_{t} is at index t-1

    def reverse(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.unet(x, t)

    # this is only for a SINGLE t (for example t ~ Cat{1, ... T})
    def forward(self, x0, t):
        # Apply forward diffusion process
        
        # we have this code, but it can be written more clearly
        xt = self.forward_diffusion(x0, self.Q_bar[t-1]) # use self.Q_bar (apply matrix mult and use basic torch sampler)
        
        # we don't have this code...
        # this is used directly in the loss
        q_tm1 = self.create_q_tm_1(x0, t) # this is just a probability distribution of some shape
        
        
        # p_theta(x_{t-1} | x_t, t)
        # this is also a probability distribution
        out = self.reverse(xt, t) # this prediction is the sample shape as q_tm1
        
        # now perform the kld divergence (this is the loss)
        kld = torch.sum( q_tm1 * torch.log(q_tm1/out))
        return kld
        # return kld (this is the loss)
        



if __name__ == '__main__':
    # Possible example
    # Set to 10 as discussed in group
    K = 10  # Number of lightness values for grayscale images
    num_timesteps = 100  # Number of time steps in the diffusion process

    in_channels = 1  # 1 for default greyscale
    out_channels = 1  # 1 for default greyscale
    num_hidden_channels = 64  # Number of hidden channels in the U-Net

    beta_values = [0.1] * num_timesteps  # Example beta values for each time step

    cat_diff_model = CategoricalDiffusionGrayscale(in_channels, out_channels, num_hidden_channels, K, num_timesteps, beta_values)
