import sys
import torch

from torch.nn.functional import softmax, log_softmax

# numerically stable kld for logits
def kld_logits(true_logits, model_logits, eps=1e-6):
    sm = softmax(true_logits + eps, dim=1)
    kld = sm * (log_softmax(true_logits + eps, dim=1) - log_softmax(model_logits + eps, dim=1))
    return kld 

# likelihood for first step of diffusion process
def cat_log_likelihood(x0, model_logits):
    log_probs = log_softmax(model_logits, dim=1)
    cat_ll = torch.sum(x0 * log_probs, dim=1)
    return cat_ll

# cross entropy for first step of diffusion process
def cat_cross_entropy(x0, model_logits):
    log_probs = log_softmax(model_logits, dim=1)
    cat_ce = torch.sum(-x0 * log_probs, dim=1)
    return cat_ce

def normalize(int_vctr, num_classes):
    return 2*(int_vctr/num_classes) - 1

def get_logits_from_logistic_pars(loc, log_scale, num_classes):
    """
    Computes logits for an underlying logistic distribution.
    Adopted from Discrete Diffusion.
    Args:
        loc (torch.Tensor): A tensor containing the location parameters of the logistic distribution.
                            Shape: (batch_size, height, width, channels)
        log_scale (torch.Tensor): A tensor containing the log scale parameters of the logistic distribution.
                                  Shape: (batch_size, height, width, channels)
        num_classes (int): Number of classes in the discrete distribution.
    Returns:
        torch.Tensor: A tensor containing the logits for the logistic distribution.
    """

    # The loc and log_scale are assumed to be modeled for data re-scaled
    # such that the values {0, ...,K-1} map to the interval [-1, 1].
    loc = loc.unsqueeze(-1)
    log_scale = log_scale.unsqueeze(-1)

    # Shift log_scale such that if it’s zero the output distribution
    # has a reasonable variance.
    inv_scale = torch.exp(-(log_scale - 2.))

    bin_width = 2. / (num_classes - 1.)
    bin_centers = torch.linspace(start=-1., stop=1., steps=num_classes)
    bin_centers = bin_centers.view(*([1] * (loc.ndim - 1)), -1)

    bin_centers = bin_centers - loc
    # Note that the edge bins corresponding to the values 0 and K-1
    # don’t get assigned all of the mass in the tails to +/- infinity.
    # So the logits correspond to unnormalized log probabilites of a
    # discretized truncated logistic distribution.
    log_cdf_min = torch.log(torch.sigmoid(inv_scale * (bin_centers - 0.5 * bin_width)))
    log_cdf_plus = torch.log(torch.sigmoid(inv_scale * (bin_centers + 0.5 * bin_width)))

    logits = log_minus_exp(log_cdf_plus, log_cdf_min)

    return logits

def log_minus_exp(a, b, epsilon=1.e-6):
    """
    Computes the log(exp(a) - exp(b)) (b<a) in a numerically stable way.
    Args:
        a (torch.Tensor): A tensor containing the a values.
        b (torch.Tensor): A tensor containing the b values.
        epsilon (float, optional): A small value to ensure numerical stability. Defaults to 1.e-6.
    Returns:
        torch.Tensor: A tensor containing the log(exp(a) - exp(b)) values.
    """
    return a + torch.log1p(-torch.exp(b - a) + epsilon)