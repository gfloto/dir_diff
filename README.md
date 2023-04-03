# Dirichlet Diffusion

## Step 1: VQ-VAE
Using the [taming transformers](https://github.com/CompVis/taming-transformers) repo.  
The repo has a terrible dependency chain. First step is to remove the required model so that we can train encoder-decoder models on any dataset. 

Primary code can be found in ``vq`` (renamed taming-transformers repo)

---

Current progress: model has been cut out, currently trying to train on celeb-a (currently balancing losses, optimization weights etc)
Information: the difficulty here is that the loss is multi-objective.  
* $l_1$ loss (pixel space)
* [perceptual loss](https://arxiv.org/abs/1801.03924)
* GAN-loss

# Score Matching
There are a number of issues with trying to formulate Dirichlet Diffusion using a Markov chain approach. For the time being, we will program using the score-matching diffusion formulation, which is much more suitable.

The important papers behind this method are the [original](https://arxiv.org/pdf/1907.05600.pdf) paper and the [more developed](https://arxiv.org/abs/2011.13456) version.

## Basics
Let $p_{\alpha_t}(x'|x) = \mathcal{D(x';\alpha_t)}$ be a pertubation kernel.  
$p_{\alpha_t}(x') = \int p_{\mathrm{data}}(x)p_{\alpha_t}(x'|x)dx$ is a noised version of our original data distribution.

We are modelling a distribution over categorical variables, meaning that our data a $t=0$ lies on corners of the probability simplex $\Sigma_d := \{x \in\mathbb{R}_{+}^{d} : x^\top \bm{1}_d = 1 \}$. In other words, our data from $p_{\mathrm{data}}(x)$ is always a one-hot vector corresponding to a probability distribution over the correct discrete category.

We are interested in learning the score of the distribution $\nabla_x\textrm{log }p_t(x)$ via the model $s_\theta(x,t)$ which can be estimated by

$$\theta^* = \mathrm{argmin}_{\theta}\left( \mathbb{E}_{t\sim U(0,1)}\mathbb{E}_{x_0\sim p_{\mathrm{data}}(x)} \mathbb{E}_{x_t\sim p(x_t|x_0)} \lambda_t \left[\Vert s_\theta(x_t,t)- \nabla_{x_t}\textrm{log }p_t(x_t|x_0) \Vert^2_2 \right] \right)$$

Dirichlet diffusion is intended to work s.t.  
$\alpha_{t=0}(x_0)$ is a one hot corresponding to $x_0$ and $\alpha_{t=1}$ is uniform.  
$p(x_t|x) \sim D(x_t | \alpha_t(x))$

## Some Problems
The results are based on the solution to an Ito sde:  
$dx = f(x,t)dt + g(t)dw$  
where the reverse sde can be written as  
$dx = [f(x,t) - g(t)^2\nabla_x\textrm{log }p_t(x)]dt + g(t)dw'$  
($w'$ is the Wiener process flowing back in time)

If we take a look at this paper on [dirichlet diffusion](https://arxiv.org/pdf/1303.0217.pdf)

## UNet - Default
Inputs 
* $x\in\mathbb{R}^{h\times w\times c}, t \in \mathbb{R}$
* $x\in\mathbb{Z}^{h\times w\times c}, t \in \mathbb{R}$

Problem: $t$ is too tiny
* $t' = \phi(t)$
* attention throughout U-net $t$
* cross-attention for patches of $x$
* eg. $x\in\mathbb{R}^{h'\times w'\times c'} \rarr x\in\mathbb{R}^{hwc}$
* $h'\ll h$

$x\in\mathbb{R}^{k\times h\times w\times c}$

TODO: how is the attention working (ie. input size etc)? 

## Ideas:
1. (b k h w c -> b h w (c k)) then do 2D convolution (stacks)
2. stack conv (ie shrink) then attention

# Dirichlet Diffusion
We would like a diffusion process with the dirichlet as the stationary distribution  
$\mathcal{D}(\bm{x}, \bm{\alpha}) = \frac{1}{B(\bm{\alpha})} \prod_{i=1}^{N} x_i^{\alpha_i -1}$

The Ito diffusion process can be written as:  
$dX_i(t) = f_i(\bm{X})dt + \sigma_{ij}(\bm{X})dW_j(t)$  
where  
$i,j = 1, \cdots, N-1$  
$X_N = 1 - \sum_i^{N-1} X_i$

Alternatively we can write this is the notation of the Fokker-Planck equation, which governs the joint probability $P(\bm{X}, t)$  
$\frac{\partial}{\partial t}P(\bm{X}, t) = -\frac{\partial}{\partial X_i}[f_i(\bm{X}) P(\bm{X}, t)] + \frac{1}{2}\frac{\partial^2}{\partial X_i X_j}[D_{ij}P(\bm{X}, t)]$  
where $D_{ij} = \sigma_{ik}\sigma_{kj}$ and summation is implied over repeated indicies  
(Note the connection between the deterministic part of this equation and the [continuity equation](https://en.wikipedia.org/wiki/Continuity_equation))

We would now like to specify the functional forms of $f_i(\bm{X})$ and $\sigma_{ij}(\bm{X})$ s.t. the stationary solution of our Fokker-Planck equation remains Dirichlet distributed. One potential solution will exist is the following is satisfied (some general sde type condition):  
$\frac{\partial}{\partial X_j}\textrm{log }P(\bm{X},t) = D^{-1}_{ij}(2f_i - \frac{\partial}{\partial X_k}D_{ik}) := \frac{\partial}{\partial X_j}\phi$  

Note that $\phi$ can be written as:  
$\phi(\bm{X}) = -\sum_{i=1}^{N}(\alpha_i - 1)\textrm{log }X_i$

Skipping the proofs, we can write the desired drift and diffuion terms as:  
$f_i(\bm{X}) = \frac{\sigma_i}{2}[\gamma_i X_N - (1 - \gamma_i)X_i]$  
$D_{ij}(\bm{X}) = \kappa_i X_i X_N$ when $i=j$, else $0$  
The coefficients must satisfy: $\sigma_i > 0, \kappa_i > 0, 0< \gamma_i < 1$  
Remember that: $X_N = 1 - \sum_i^{N-1} X_i$

Furthermore, the stationary distribution has the following parameters:  
$\alpha_i = \frac{\sigma_i}{\kappa_i}\gamma_i$ where $\alpha = 1, \cdots, N-1$  
$\alpha_N = \frac{\sigma_i}{\kappa_i}(1 - \gamma_i)$ for $i = 1, \cdots, N-1$
