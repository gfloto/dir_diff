# Score Matching on the Probability Simplex
To perform score matching on the probability simplex we need the gradient of the log-logit Gaussian distribution

$$ p(x) = \frac{1}{\sqrt{2\pi v}} \frac{1}{x(1-x)} \textrm{exp }\left(-\frac{(\sigma^{-1}(x) - \mu)^2}{2v} \right)$$

where $\sigma^{-1} = \textrm{log }\left( \frac{x}{1-x} \right)$

## Extension to larger Domains
We will find it useful to slightly generalize this to be a distribution over any simplex of length $a$. For now, we will work in the 1D case.

Previously, we used $\sigma (x) = \frac{1}{1+e^{-x}}$ to map points from $\mathbb{R}$ to $(0,1)$. We can use $\sigma_a(x) = \frac{a}{1+e^{-x}}$ where points are mapped from $\mathbb{R}$ to $(0,a)$. To get the corresponding probability distribution, we use the change of variables formula

$$p_a(x) = p(\sigma_a^{-1}(x)) \frac{\partial}{\partial x} \sigma_a^{-1}(x)$$

where is this case, $p(x)$ is the standard Gaussian. Note that $\sigma_a^{-1}(x) = \textrm{log }\left[\frac{x}{a-x}\right]$ We just need to compute 

$$
\begin{aligned}
    \frac{\partial}{\partial x} \sigma_a^{-1}(x) &= \frac{\partial}{\partial x} \textrm{log }\left[\frac{x}{a-x}\right] \\
    &= \frac{a}{x(a-x)}
\end{aligned}
$$

meaning that we can write a slightly more general logit-normal distribution as:

$$ p_a(x) = \frac{1}{\sqrt{2\pi v}} \frac{a}{x(a-x)} \textrm{exp }\left(-\frac{(\sigma_a^{-1}(x) - \mu)^2}{2v} \right)$$


## Score derivation in 1D

We are interested in:  
$\nabla_x \textrm{log }p(x)$  
or for the time being:  
$\frac{\partial}{\partial x} \textrm{log }p(x)$  

After working in 1D, we will then show the general case.  

First we deal with the log prob:  

$$
\begin{aligned}
    \frac{\partial}{\partial x} \textrm{log }p_a(x) &= \textrm{log }\left[ \frac{1}{\sqrt{2\pi v}} \frac{a}{x(a-x)} \textrm{exp }\left(-\frac{(\sigma_a^{-1}(x) - \mu)^2}{2v} \right) \right]\\
    &= C + \textrm{log }\left[ \frac{a}{x(a-x)} \right] - \frac{(\sigma_a^{-1}(x) - \mu)^2}{2v}
\end{aligned}
$$

where $C = \textrm{log }[\frac{1}{\sqrt{2\pi v}}]$  

Next, we can then differentiate each of the components seperately

The first can be solved as
$$
\begin{aligned}
    \frac{\partial}{\partial x} \textrm{log }\left[ \frac{a}{x(a-x)} \right] &= -\frac{\partial}{\partial x} \textrm{log } [x] -\frac{\partial}{\partial x}\textrm{log }[a-x] \\
    &= -\frac{1}{x} +\frac{1}{a-x}
\end{aligned}
$$

and the second can be solved as

$$
\begin{aligned}
    \frac{\partial}{\partial x} -\frac{(\sigma_a^{-1}(x) - \mu)^2}{2v} &= -\frac{1}{2v}\frac{\partial}{\partial x} \left(\sigma_a^{-1}(x) - \mu\right)^2 \\
    &= -\frac{\sigma_a^{-1}(x) - \mu}{v} \frac{\partial}{\partial x} \left( \textrm{log } \left[ \frac{x}{a-x} \right] - \mu \right) \\
    &= -\frac{\sigma_a^{-1}(x) - \mu}{v} \frac{a}{x(a-x)} \\
    &= -\frac{a\sigma_a^{-1}(x) - a\mu}{vx(a-x)}
\end{aligned}
$$

Putting it all together, we get:

$$
\begin{aligned}
    \frac{\partial}{\partial x} \textrm{log }p_a(x) &= \frac{1}{a-x} -\frac{a}{x} -\frac{a\sigma_a^{-1}(x) - a\mu}{vx(a-x)} \\
    &= -\frac{a\sigma_a^{-1}(x) -2vx - a\mu + av}{vx(a-x)} \\
    &= \frac{2vx + a\mu -a\sigma_a^{-1}(x) -av}{vx(a-x)} \\
\end{aligned}
$$

# General Case
The general logit-Guassian distribution can be written as:  
$$ p(x) = \frac{1}{Z} \frac{1}{\prod^d_{i=1} x_i} \textrm{exp }\left(-\frac{\Vert\textrm{log }\left[\frac{\bar{x}_d}{x_d} \right] - \mu\Vert_2^2}{2v} \right)$$

where $x\in\mathcal{S}^d$ and $\bar{x}_d = [x_1,\dots,x_{d-1}]$.  
We assume that the Gaussian has covariance $\Sigma = \sqrt{v}I$

To bring $x$ from the simplex back to $\mathbb{R}^{d-1}$ we can use:  
$$y_i = \textrm{log }\left[\frac{x_i}{x_d}\right], i\in \{1,\dots,d-1\}$$

The inverse transformation of this is:  
$$x_i = \frac{e^{y_i}}{1+\sum_{k=1}^{d-1}e^{y_k}}, i\in \{1,\dots,d-1\}$$
$$x_d = \frac{1}{1+\sum_{k=1}^{d-1}e^{y_k}} = 1 - \sum_{i=1}^{d-1}x_i$$

## Extension to Larger Domain
Yet again, we are interested in extending this definition to work for large domains.

First we must find the function to map from $\mathbb{R}^d$ to $\mathcal{S}^d_a$, which is the simplex of length $a$. To begin, we define

$$
\begin{aligned}
    x_i &= \sigma_a(y_i) = a\sigma(y_i), i\in \{1,\dots,d-1\}\\ 
    x_d &= a - \sum_{i=1}^{d-1}x_i 
\end{aligned}
$$

where $\sigma$ is the previous tranformation from $\mathbb{R}^d$ to $\mathcal{S}^d_a$. The inverse can easily be seen to be the same as the previous example, meaning that:

$$\sigma^{-1}_a(\bm{x}) = \sigma^{-1}(\bm{x})$$

Working in more dimensions, the change of variables formula is:

$$p_a(\bm{x}) = p(\sigma_a^{-1}(\bm{x})) \bigg\vert \textrm{det }\frac{\partial}{\partial \bm{x}} \sigma_a^{-1}(\bm{x}) \bigg\vert$$

Thus, our next step is to find this log det term. We shall do so by first getting the Jacobian matrix into a convenient form.

$$
\begin{aligned}
    \frac{\partial}{\partial x_j} \sigma_a^{-1}(x)_i &= \frac{\partial}{\partial x_j} \textrm{log }\left[\frac{x_i}{x_d}\right] \\
    &= \frac{\partial}{\partial x_j} \textrm{log }[x_i] - \frac{\partial}{\partial x_j} \textrm{log }\left[a - \sum_{i=1}^{d-1}x_i \right] \\
    &= \delta_{ij}\frac{1}{x_i} - \frac{\partial}{\partial u} \textrm{log }[u] \frac{\partial}{\partial x_j}u, u =  a - \sum_{i=1}^{d-1}x_i\\
    &= \delta_{ij}\frac{1}{x_i} + \frac{1}{x_d}\\
\end{aligned}
$$

Now that we have the Jacobian, we can calculate the determinant by using the fact that a diagonal matrix $D$ plus a constant matrix $C$ has the following determinant:

$$\textrm{det }(D+C) = \left(1+c\sum_{i=1}^{n}d_i^{-1}\right)\prod_{i=1}^n d_i$$

In our case we have the following:

$$
\begin{aligned}
    \bigg\vert \textrm{det }\frac{\partial}{\partial \bm{x}} \sigma_a^{-1}(\bm{x}) \bigg\vert &= \left(1+\frac{1}{x_d}\sum_{i=1}^{d-1}x_i\right)\prod_{i=1}^{d-1}\frac{1}{x_i} \\
    &= \left(1+\frac{1}{x_d}(a-x_d)\right)\prod_{i=1}^{d-1}\frac{1}{x_i} \\
    &= \prod_{i=1}^{d}\frac{a}{x_i} \\
\end{aligned}
$$

We now have all the ingredients to make our slightly more general logit-Gaussian distribution:

$$p_a(x) = \frac{1}{(2\pi)^{(d-1)/2}} \frac{a}{\prod_{i-1}^{d}x_i} \textrm{exp }\left(-\frac{1}{2v}\bigg\Vert\textrm{log }\left[\frac{\bar{x}_d}{x_d} \right] - \mu \bigg\Vert_2^2 \right)$$

---

## Derivation of Score

Overall, we want to calculate:  
$\nabla_x\textrm{log }p_a(x)$  

Following the same process as the 1D case:  
$$\textrm{log }p_a(x) = -\textrm{log }[Z] - \textrm{log }\left[\prod_{i-1}^{d}x_i\right] -\frac{1}{2v}\bigg\Vert\textrm{log }\left[\frac{\bar{x}_d}{x_d} \right] - \mu \bigg\Vert_2^2$$ 

We deal with the gradients, starting with the second term (the first one has no gradient). 

$$
\begin{aligned}
    g &:= -\nabla_x\textrm{log }\left[\prod_{i=1}^{d}x_i \right] \\ 
    g_i &= -\frac{\partial}{\partial x_i}\left(\sum_{i=1}^{d-1}\textrm{log }[x_i] + \textrm{log }\left[a - \sum_{k=1}^{d-1}x_k \right]\right) \\
    &= -\frac{1}{x_i} + \frac{1}{a - \sum_{k=1}^{d-1}x_k} \\
    &= \frac{1}{x_d} - \frac{1}{x_i} \\
    &= \frac{x_i - x_d}{x_ix_d} \\
\end{aligned}
$$

Next, we deal with the exponential term:

$$
\begin{aligned}
    h &:= -\frac{1}{2v}\nabla_x \bigg\Vert\textrm{log }\left[\frac{\bar{x}_d}{x_d} \right] - \mu \bigg\Vert_2^2 \\ 
    h_i &= -\frac{1}{2v}\frac{\partial}{\partial x_i}\left(\sum_{k=1}^{d-1}\left(\textrm{log }\left[\frac{x_k}{x_d} \right]-\mu \right)^2 \right) \\
    &= -\frac{1}{2v}\sum_{k=1}^{d-1}\left(\frac{\partial}{\partial u}u^2\frac{\partial}{\partial x_i}u \right), u = \textrm{log }\left[\frac{x_k}{x_d} \right]-\mu \\
\end{aligned}
$$

We can just focus on $\beta := \frac{\partial}{\partial u}u^2\frac{\partial}{\partial x_i}u$ for now

$$
\begin{aligned}
    \beta &:= \frac{\partial}{\partial u}u^2\frac{\partial}{\partial x_i}u \\
    &= 2u \left(\frac{\partial}{\partial x_i}\textrm{log }[x_k] - \frac{\partial}{\partial x_i}\textrm{log} \left[a - \sum_{k=1}^{d-1}x_k \right]\right) \\
    &= 2u \left(\delta_{ik}\frac{1}{x_i} + \frac{1}{x_d}\right) \\
\end{aligned}
$$

Combining terms we get:

$$
\begin{aligned}
    h_i &= -\frac{1}{v}\sum_{k=1}^{d-1}\left(\delta_{ik}\frac{1}{x_i} + \frac{1}{x_d}\right)\left(\textrm{log }\left[\frac{\bar{x}_d}{x_d} \right]-\mu\right) \\
    &= -\frac{1}{vx_d}\sum_{k=1}^{d-1}\left(\textrm{log}\left[\frac{x_k}{x_d}\right]-\mu \right) - \frac{1}{vx_i}\left(\textrm{log}\left[\frac{x_i}{x_d}\right]-\mu \right) \\
    &= -\frac{1}{vx_d}\sum_{k=1}^{d-1}\gamma_\mu^k(\bm{x}) - \frac{1}{vx_i}\gamma_\mu^i(\bm{x}) \\
\end{aligned}
$$

where we write $\gamma_\mu^i(\bm{x}) = \textrm{log}\left[\frac{x_i}{x_d}\right]-\mu$

For the final results, we must combine the $h$ and $g$ terms together to get:

$$
\begin{aligned}
    \textrm{log }p_a(x)_i = -\frac{1}{vx_d}\sum_{k=1}^{d-1}\gamma_\mu^k(\bm{x}) - \frac{1}{vx_i}\gamma_\mu^i(\bm{x}) + \frac{x_i - x_d}{x_ix_d} \\
\end{aligned}
$$