import torch
import torch.nn as nn
import torch.distributions as dist

# import the unet model 
# from unet import UNet


class CategoricalDiffusionGrayscale(nn.Module):
    def __init__(self, in_channels, out_channels, num_hidden_channels, K, num_timesteps, beta_values):
        super(CategoricalDiffusionGrayscale, self).__init__()

        # U-Net for predicting noise schedule
        # Subject to change 
        self.unet = UNet(in_channels, out_channels, num_hidden_channels)
        
        self.tau = 1.0  # Temperature parameter, adjust as needed
        self.K = K
        self.num_timesteps = num_timesteps
        self.beta_values = beta_values
        self.ones = torch.ones(K, K)

    # Compute the Q_t matrix for a given timestep t
    # using the Uniform distribution defined in Appendix A.2.1
    def compute_Q_t(self, t):
        beta_t = self.beta_values[t]
        Q_t = (1 - beta_t) * torch.eye(self.K) + beta_t * self.ones / self.K
        return Q_t

    # Add noise to the image x0 using the forward diffusion process
    def forward_diffusion(self, x0, t):
        # this is just the matrix multiplication of Q_t
        bar_Q_t = torch.eye(self.K)
        for i in range(t):
            bar_Q_t = bar_Q_t @ self.compute_Q_t(i)
        p = torch.matmul(x0, bar_Q_t)
    
        # Gumbel-Softmax reparameterization
        logits = torch.log(p)
        x_t = self._gumbel_softmax_sample(logits, self.tau)
    
        return x_t

    def reverse(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.unet(x, t)

    def forward(self, x0, t):
        # Apply forward diffusion process
        return self.forward_diffusion(x0, t)
    
    def _gumbel_softmax_sample(self, logits, tau):
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
        y = logits + gumbel_noise
        return torch.nn.functional.softmax(y / tau, dim=-1)


if __name__ == '__main__':
    # Possible example
    # Set to 10 as discussed in group
    K = 10  # Number of lightness values for grayscale images
    num_timesteps = 100  # Number of time steps in the diffusion process

    in_channels = 1  # 1 for grayscale
    out_channels = 1  # 1 for grayscale
    num_hidden_channels = 64  # Number of hidden channels in the U-Net

    beta_values = [0.1] * num_timesteps  # Example beta values for each time step

    cat_diff_model = CategoricalDiffusionGrayscale(in_channels, out_channels, num_hidden_channels, K, num_timesteps, beta_values)
