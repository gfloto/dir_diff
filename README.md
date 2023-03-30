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