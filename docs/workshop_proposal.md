# Lifting Discrete Diffusion to Continuous Spaces
This document is intended to outline the way I think our work should be framed for the upcoming workshop paper. 

## Motivation
Discrete denoising diffusion models (D3M) scales poorly with the number of categories. D3M works by representing categories as one hot vectors, that are noised in discrete time steps by some transition matrix $Q_t$ s.t. $q(x_t | x_{t-1}) = Q_tx_{t-1}$ where $x$ is one-hot (see [here](https://arxiv.org/pdf/2107.03006.pdf) for more details). 

There are two problems with this method:

1. The predicted denoising $p_\theta(x_{t-1} | x_t)$ is in $\mathbb{R}^k$ where $k$ is the number of categories
2. Storing the transition matricies $Q_t$ scales as $\mathcal{O}(Tk^2)$ where $T$ is the number of time steps  
    * There are some ways to reduce this

As an example, LLM's use on the order of ten thousand tokens. In this case $p_\theta(x_{t-1} | x_t)$ would need to be modelled by a neural network that has input and output of $k = 10 000$ which is inpractical to impliment.


## Our Proposed Method
By "lifting" the discrete diffusion process to a continuous space, we can reduce the model output to $\textrm{log}_2(k)$ and remove the need to store large transition matricies $Q_T$

We also show that a by-product of our methodology allows us to naturally extend to [simplex diffusion](https://arxiv.org/abs/2210.14784) in a simple and easy to implement manner. 

---

## Method
To resolve the scaling problems of D3M, we choose to lift the discrete data to a continuous domain where the use of score-matching gives us greater flexibility. Categorical data can be represented as binary strings, which are geometrically the corners of a hyper-cube. Our approach is to apply the score matching methodology to this more convenient representation that has a much smaller dimensionality of $\textrm{log}_2(k)$. 


