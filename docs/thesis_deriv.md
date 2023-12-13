$$
    \newcommand{\x}{\mathbf{x}}
    \newcommand{\z}{\mathbf{z}}
    \newcommand{\w}{\mathbf{w}}
    \newcommand{\f}{\mathbf{f}}
    \newcommand{\G}{\mathbf{G}}
    \newcommand{\a}{\mathbf{a}}
    \newcommand{\b}{\mathbf{b}}
    \newcommand{\d}{\mathrm{d}}
    \newcommand{\1}{\mathbf{1}}
    \newcommand{\half}{\frac{1}{2}}
$$


# Forward Process

## Pushing Gaussian Diffusion on the Probability Simplex
We would like to define a diffusion process on the unit simplex. Also known as the probability simplex, this is the set of points that sum to 1 and are positive: 

$$
    \mathbb{S}^d = \left\{ \x \in \mathbb{R}^d \ \middle | \ \sum_{k=1}^d x_k = 1, \ x_k \geq 0 \right\}.
$$

To define a diffusion process on the simplex, we will use the commonly used Gaussian diffusion and push it forward to the simplex $\mathbb{S}^d$ Typically, diffusion is defined as the following OU process on $\mathbb{R}^d$:

$$
    \d\z = -\frac{1}{2}\beta(t) \z \ \d t + \sqrt{\beta(t)} \ \d\w
$$

where the function $\beta(t)$ is a user-selection function that controls the rate that noise is added to the data sample, and $\w$ is a standard Brownian motion. We would like to push the process forward onto the unit simplex in the following manner: 

$$
    \d\x := \sigma(\z).
$$

At this point, it is worth noting that $\mathbb{S}^d$ is a $d-1$ dimensional subset of $\mathbb{R}^d$, given that the final component of vectors on the simplex can be written as $\x_d = 1 - \sum_{k=1}^{d-1} \x_k$. We can then consider the simplex as being fully determined by the set of points that are positive and sum to $\textit{less than or equal}$ to 1: 

$$
    \left\{\x \in \mathbb{R}^{d-1} \ \middle | \ \sum_{k=1}^{d-1} \x_k \leq 1, \ x_k \geq 0 \right\}.
$$

Due to this property, we would like $\sigma(\z)$ to map vectors in $\mathbb{R}^{d-1}$ to the first $d-1$ dimensions of the simplex, and the final component will be fully determined as $1 - \sum_{k=1}^{d-1} \sigma_k(\z)$. To achieve this, we can use $\sigma(\z) : \mathbb{R}^{d-1} \rightarrow \mathbb{S}^d$ defined as:

$$
    \sigma(\z) = \frac{e^{\z}}{1 + \sum_{k=1}^{d-1} e^{\z_k}}.
$$

We will use the convention where only the first $(d-1)$ components of $\z$ are written, with the understanding that the last component it always fully determined. It will be useful to have the inverse of this function as well, which is defined as:

$$
    \sigma^{-1}(\x) = \log\left( \frac{\x}{1 - \|\x\|_1} \right).
$$

## Deriving the SDE with Ito's Lemma
As described in some other section, we would like our SDE to be in the following form to be able to use the score matching method:

$$
    \d\x = \f_t(\x) \ \d t + \G_t(\x) \ \d\w.
$$

To derive this form, we will use Ito's Lemma to find the SDE for $\x$:

$$
    \d\x_i = \sigma_i(\z) = \left\{
        -\half\beta(t) (\nabla_\z \sigma_i(\z))^\top \z
        + \half\beta(t) \ \mathrm{Tr}\left[ H_z \sigma_i(\z) \right]
    \right\} \ \d t
    + \sqrt{\beta(t)} (\nabla_\z \sigma_i(\z))^\top \ \d\w
$$

where $H_z$ is the Hessian of $\sigma(\z)_i$ with respect to $\z$. To simplify this equation, we will first work with the gradient of the i-th component of $\sigma(\z)$ and then the Hessian.

### Gradient of $\sigma_i(\z)$ leading to Diffusion Matrix Term

$$\begin{aligned}
    \nabla_\z \sigma_i(\z) &= \nabla_\z \frac{e^{\z_i}}{1 + \sum_{k=1}^{d-1} e^{\z_k}} \\ 
    (\nabla_\z \sigma_i(\z))_j &= \frac{\partial}{\partial \z_j} \frac{e^{\z_i}}{1 + \sum_{k=1}^{d-1} e^{\z_k}} \\
\end{aligned}$$

There are two different cases to consider, when $j=i$ and when $j\neq i$. We consider the first of these below. For convenience, we will use $\alpha(\z) = 1 + \sum_{k=1}^{d-1}e^{\z_k}$.

$$\begin{aligned}
    (\nabla_\z \sigma_i(\z))_i &= \frac{\partial}{\partial \z_i} \frac{e^{\z_i}}{\alpha(\z)} \\
    &= \alpha(\z)^{-2} \left[ \alpha(\z) \frac{\partial}{\partial \z_i} e^{\z_i} - e^{\z_i} \frac{\partial}{\partial \z_i} \alpha(\z) \right] \\
    &= \alpha(\z)^{-2} \left[ e^{\z_i} \alpha(\z) - e^{2\z_i} \right] \\
    &= e^{\z_i} \alpha(\z)^{-1} \left[ \alpha(\z) - e^{\z_i} \right] \\
    &= \sigma_i(\z) \left[ 1 - \sigma_i(\z) \right] \\
    &= \x_i (1 - \x_i) \\
\end{aligned}$$

Next, the case when $j\neq i$:

$$\begin{aligned}
    (\nabla_\z \sigma_i(\z))_j
    &= \frac{\partial}{\partial \z_j} \frac{e^{\z_i}}{\alpha(\z)} \\
    &= -\alpha(\z)^{-2} \frac{\partial}{\partial \z_j} \alpha(\z) e^{\z_i} \\
    &= -\alpha(\z)^{-2} e^{\z_i} e^{\z_j} \\
    &= -\sigma_i(\z) \sigma_j(\z) \\
    &= -\x_i \x_j \\
\end{aligned}$$

At this point, we can notice that $G_t(\x)_i = \sqrt{\beta(t)} \nabla_\z \sigma_i(\z)^\top$ and can write out the full diffusion matrix $G_t(\x)$ as:

$$\begin{aligned}
    G_t(\x) 
    &= \sqrt{\beta(t)} J_\z \sigma(\z) \\
    (J_\z \sigma(\z))_{i,j}
    &= \begin{cases}
        \x_i (1 - \x_i) & \textrm{if } \ \ i=j \\
        -\x_i \x_j & \textrm{if } \ \ i\neq j \\
    \end{cases} \\
\end{aligned}$$


### Hessian of $\sigma_i(\z)$ leading to Drift Term

Next we deal with the trace Hessian term:

$$\textrm{Tr}[H_{X}\sigma_i(\mathbf{\z})] = \sum_{j=1}^{d-1}\frac{\partial^2}{\partial \z_j^2} \sigma_i(\mathbf{X})_j$$

which again can be split into two cases. First we deal with the case when $j=i$

$$\begin{aligned}
    \frac{\partial^2}{\partial \z_i^2} \sigma_i(\z)
    &= \frac{\partial}{\partial \z_i} \sigma_i(\z)(1-\sigma_i(\z)) \\
    &= \sigma_i(\z)(1-\sigma_i(\z))(1-2\sigma_i(\z)) \\
    &= \x_i(1-\x_i)(1-2\x_i)
\end{aligned}$$

Then the case where $j\neq i$

$$\begin{aligned}
    \frac{\partial^2}{\partial \z_j^2} \sigma_i(\z)
    &= -\frac{\partial}{\partial \z_j} \sigma_i(\z)\sigma_j(\z)\\
    &= -\sigma_i(\z)\sigma_j(\z)(1 - 2\sigma_j(\z)) \\
    &= -\x_i\x_j(1-2\x_j)
\end{aligned}$$

We can now combine these terms together and simplify:

$$\begin{aligned}
    \mathrm{Tr}[H_{X}\sigma_i(\mathbf{\z})]
    &= \x_i(1 - \x_i)(1 - 2\x_i) + \sum_{j\neq i} -\x_i\x_j(1-2\x_j) \\
    &= \x_i(1 - \x_i)(1 - 2\x_i) - \x_i \left[
        -\x_i(1-2\x_i) + \sum_j \x_j(1-2\x_j)
    \right] \\
    \mathrm{Tr}[H_{X}\sigma(\mathbf{\z})]
    &= \x(\1-\x)(\1-2\x) + \x^2(\1-2\x) - \|\x(\1-2\x)\|_1 \ \x \\
    &= \x(\1-2\x) - \|\x(\1-2\x)\|_1 \ \x \\
    &= -2\x^2 + \x - \|\x(1-2\x)\|_1 \ \x \\
\end{aligned}$$

Now we could like to simplify terms to get the drift term $\f_t(\x)$. We must work with the following:

$$\begin{aligned}
    \f_t(\x)_i
    &= -\half\beta(t) (\nabla_\z \sigma_i(\z))^\top \z
        + \half\beta(t) \ \mathrm{Tr}\left[ H_z \sigma_i(\z) \right] \\
    &= \half \beta(t) \left\{
        \mathrm{Tr}\left[ H_z \sigma_i(\z) \right]
        - (\nabla_\z \sigma_i(\z))^\top \z
    \right\} \\
    \f_t(\x)
    &= -\half \beta(t) \left[
        2\x^2 - \x + \|\x(1-2\x)\|_1 \ \x
        + \frac{1}{\sqrt{\beta(t)}} \G(\x) \sigma(\x)^{-1}
    \right]

\end{aligned}$$

# Reverse Process
The reverse diffusion process is: 

$$
    \d\x = \left\{
        \f_t(\x) 
        - \nabla_\x \cdot [\G_t(\x) \G_t(\x)^\top]
        - \G_t(\x) \G_t(\x)^\top \nabla_\x \log p_t(\x)
    \right\} \d t \
    + \G_t(\x) \ \d\w
$$

where (insert definition of matrix divergence) we are able to use the fact that $\G_t(\x)^\top = \G_t(\x)$. We will also drop the dependence on $t$ for visual clarity. 

## Diffusion Matrix Divergence Term
First, we will simplify the divergence term. To do this, we will begin with  the following:


$$
    \nabla_\x \cdot [\G_t(\x) \G_t(\x)^\top]_i
    = \sum_j \frac{\partial}{\partial \x_j} [\G_t(\x) \G_t(\x)^\top]_{i,j} \\
$$

where we will again split the summation into two cases, when $i=j$ and when $i\neq j$. We begin with the case when $i=j$, where the diffusion matrix is first expanded:

$$\begin{aligned}
    \G(\x)^2_{i,i}
    &= \sum_k \G(\x)_{i,k} \G(\x)_{k,i} \\
    &= \G_{i,i}^2 + \sum_{k\neq i} \G_{i,k} \G_{k,i} \\
    &= \beta\x_i^2(1-\x_i)^2 + \beta(t)\x_i^2 \sum_{k\neq i} \x_k^2 \\
    &= \beta\x_i^2 \left[ (1-\x_i)^2 + \sum_{k\neq i} \x_k^2 \right] \\
\end{aligned}$$

and then the derivative computed:

$$\begin{aligned}
    \frac{\partial}{\partial \x_i} \G(\x)^2_{i,i}
    &= \beta \frac{\partial}{\partial \x_i} \x_i^2 \left[ (1-\x_i)^2 + \sum_{k\neq i} \x_k^2 \right] \\
    &= 2 \beta \x_i \left[ (1-\x_i)^2 + \sum_{k\neq i} \x_k^2 \right] - 2 \beta \x_i^2 (1-\x_i) \\
    &= 2 \beta \x_i \left[ (1 - \x_i) (1 - 2\x_i) + \sum_{k \neq i} x_k^2 \right] \\
\end{aligned}$$

Next, the case when $i\neq j$. Again, we begin by expanding the diffusion matrix:

$$\begin{aligned}
    \G(\x)^2_{i,j}
    &= \G_{i,i}\G_{i,j} + \G_{i,j}\G_{j,j} + \sum_{k\neq i,j} \G_{i,k} \G_{k,j} \\
    &= \beta \left[ 
        -\x_i^2\x_j(1-\x_i) - \x_j^2\x_i(1-\x_j) + \x_i\x_j \sum_{k\neq i,j} \x_k^2
    \right] \\
    &= -\beta \x_i\x_j \left[
        \x_i(1-\x_i) + \x_j(1-\x_j) - \sum_{k\neq i,j} \x_k^2
    \right] \\
\end{aligned}$$

and then the sum of derivative terms:

$$
    \sum_{j \neq i}\frac{\partial}{\partial \x_j} \G(\x)^2_{i,j}
    = \beta \sum_{j \neq i} \frac{\partial}{\partial \x_j} -\x_i\x_j \left[
        \underbrace{\x_i(1-\x_i) + \x_j(1-\x_j) - \sum_{k\neq i,j} \x^2_k}_{\a(\x)}
    \right] \\
$$

we can approach this by using the product rule. First we will compute the derivative of $\a(\x)$:

$$\begin{aligned}
    \frac{\partial}{\partial \x_j} \a(\x)
    &= \frac{\partial}{\partial \x_j} \left[
        \x_i(1-\x_i) + \x_j(1-\x_j) - \sum_{k\neq i,j} \x^2_k
    \right] \\
    &= (1-2\x_j) \\
\end{aligned}$$

and then the derivative of the product:

$$\begin{aligned}
    \sum_{j \neq i}\frac{\partial}{\partial \x_j} \G(\x)^2_{i,j}
    &= \beta \sum_{j \neq i} \frac{\partial}{\partial \x_j} -\x_i\x_j \a(\x) \\
    &= \beta \sum_{j \neq i} [ -\x_i\a(\x) - \x_i\x_j (1-2\x_j) ] \\
    &= -\beta \x_i \sum_{j \neq i} \left[
        \x_i(1-\x_i) + \x_j(1-\x_j) + \x_j(1 - 2\x_j) - \sum_{k\neq i,j} \x^2_k
    \right] \\
    &= -(d-2)\beta\x_i^2 (1-\x_i) - \beta\x_i \sum_{j \neq i} \left[
        \x_j(1-\x_j) + \x_j(1 - 2\x_j) - \sum_{k\neq i,j} \x^2_k
    \right] \\
    &= -(d-2)\beta\x_i^2 (1-\x_i) - \beta\x_i \sum_{j \neq i} \left[
        \x_j(2 - 3\x_j) - \sum_{k\neq i,j} \x^2_k
    \right] \\
\end{aligned}$$

Now we can combine the two cases together to get the full divergence term:

$$\begin{aligned}
    \nabla_\x \cdot [\G_t(\x) \G_t(\x)^\top]_i
    &= \sum_j \frac{\partial}{\partial \x_j} [\G_t(\x) \G_t(\x)^\top]_{i,j} \\
    &= \underbrace{2 \beta \x_i \left[ (1 - \x_i) (1 - 2\x_i) + \sum_{k \neq i} x_k^2 \right]}_{\b_1(\x)_i}
    - \underbrace{(d-2)\beta\x_i^2 (1-\x_i)}_{\b_2(\x)_i}
    - \underbrace{\beta\x_i \sum_{j \neq i} \left[
        \x_j(2 - 3\x_j) - \sum_{k\neq i,j} \x^2_k
    \right]}_{\b_3(\x)_i} \\
\end{aligned}$$

To complete this section, we would like to vectorize the equation that we have just derived. We will work on each part of the equation separately. First, we will consider the term $\b_1(\x)_i$:

$$\begin{aligned}
    \b_1(\x)_i
    &= 2 \beta \x_i \left[ (1 - \x_i) (1 - 2\x_i) + \sum_{k \neq i} x_k^2 \right] \\
    &= 2 \beta \x_i \left[
        1 - 3\x_i + 2\x_i^2 - \x_i^2 + \sum_{k} x_k^2
    \right] \\
    \b_1(\x)
    &= 2 \beta \x [\x^2 - 3\x + (1 + \|x\|_2^2)\1] \\
\end{aligned}$$

The next term can be vectorized as:

$$\begin{aligned}
    \b_2(\x)_i &= (d-2)\beta\x_i^2 (1-\x_i) \\
    \b_2(\x) &= (d-2)\beta\x^2 (\1-\x) \\
\end{aligned}$$

and the final term as:

$$\begin{aligned}
    \b_3(\x)_i
    &= \beta\x_i \sum_{j \neq i} \left[
        \x_j(2 - 3\x_j) - \sum_{k\neq i,j} \x^2_k
    \right] \\
   &= \beta\x_i \sum_{j \neq i} [ \x_j(2 - 3\x_j) + \x_i^2 + \x_j^2 - \|\x\|_2^2 ] \\
   &= (d-2)\beta\x_i(\x_i^2 - \|\x\|_2^2) 
   + \beta\x_i \sum_{j \neq i} [ \x_j(2 - 3\x_j) - \x_j^2 ] \\
   &= (d-2)\beta\x_i(\x_i^2 - \|\x\|_2^2) 
   + \beta\x_i(2-3\x_i) \left[ \sum_{j \neq i} \x_j \right]
   + \beta\x_i \left[ \sum_{j \neq i} \x_j^2 \right] \\
   &= (d-2)\beta\x_i(\x_i^2 - \|\x\|_2^2)
   + \beta\x_i(2-3\x_i) ( \|\x\|_1 - \x_i )
   + \beta\x_i (\|\x\|_2^2 - \x_i) \\
   &= (d-2)\beta\x_i(\x_i^2 - \|\x\|_2^2)
   + \beta\x_i ( 2\|\x\|_1 - 2\x_i - 3\|\x\|_1\x_i + 3\x_i^2 + \|\x\|_2^2 - \x_i ) \\
    &= (d-2)\beta\x_i(\x_i^2 - \|\x\|_2^2)
    + \beta\x_i ( 3\x_i^2 - 3(\|\x\|_1 + 1)\x_i + 2\|\x\|_1 + \|\x\|_2^2 ) \\
    \b_3(\x)
    &= (d-2)\beta\x(\x^2 - \|\x\|_2^2\1)
    + \beta\x [ 3\x^2 - 3(\|\x\|_1 + 1)\x + (2\|\x\|_1 + \|\x\|_2^2)\1 ] \\
\end{aligned}$$

Finally, we can combine these terms together to get the full divergence term:

$$\begin{aligned}
    \nabla_\x \cdot [\G_t(\x) \G_t(\x)^\top]
    &= \beta\x \left[
        (2\x^2 - 6\x + 2\|\x\|_2^2)\1
        - (d-2)\x(\1 - \x)
        - (d-2)(\x^2 - \|\x\|_2^2\1)
        - (3\x^2 - 3(\|\x\|_1 + 1)\x + (2\|\x\|_1 + \|\x\|_2^2)\1)
    \right] \\
    &= \beta\x \left[
        (2 -2(d-2) - 3)\x^2 - (6 +(d-2) + 3(\|\x\|_1 - 1))\x + (2\|\x\|_2^2 + (d-2)\|\x\|_2^2 - 2\|\x\|_1 - \|\x\|_2^2) \1
    \right] \\
    &= \beta\x \left[
        (3-2d)\x^2 - (3\|\x\|_1 + d + 1)\x + ((d-1)\|\x\|_2^2 - 2\|\x\|_1) \1
    \right]
\end{aligned}$$

## Score Derivation

The final term that we need to derive is the score term. We will begin by writing out the full equation for the score:

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
    &= -\frac{1}{vx_d}\sum_{k=1}^{d-1}\gamma_\mu^k(\mathbf{x}) - \frac{1}{vx_i}\gamma_\mu^i(\mathbf{x}) \\
\end{aligned}
$$

where we write $\gamma_\mu^i(\mathbf{x}) = \textrm{log}\left[\frac{x_i}{x_d}\right]-\mu$

For the final results, we must combine the $h$ and $g$ terms together to get:

$$
\begin{aligned}
    \nabla_x\textrm{log }p_a(x)_i = -\frac{1}{vx_d}\sum_{k=1}^{d-1}\gamma_\mu^k(\mathbf{x}) - \frac{1}{vx_i}\gamma_\mu^i(\mathbf{x}) + \frac{x_i - x_d}{x_ix_d} \\
\end{aligned}
$$

which can be vectorized as:

$$
    \nabla_\x\textrm{log }p(\x) 
    = \frac{\x - ||\x||_1 \1}{||\x||_1 \x} 
    - \frac{1}{v||\x||_1} \gamma(\x)
    -\frac{\1^\top [\gamma(\x) - \mu]}{v ||\x||_1} \1
$$
