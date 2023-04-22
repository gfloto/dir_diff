# Simplex Diffusion

A general overview of the project can be found in the report in `docs/report.pdf`

The CSC413/2516 class report can be found along with organized results in `docs/class/`

# Important Papers

This section includes relevant papers. Please include anything relevant here.

## Simplex Diffusion Relevent

Papers that are directly relevant for building a functioning model

-   [Simplex Diffusion](https://arxiv.org/abs/2210.14784)

-   Score based generative models - [website](https://yang-song.net/blog/2021/score/) \| [paper](https://arxiv.org/abs/2011.13456)

-   DDPM - [paper](https://arxiv.org/abs/2006.11239)

## Comparison Papers

Works that we may want to compare to

-   Categorical diffusion - [paper](https://arxiv.org/abs/2107.03006)

-   LM diffusion - [paper](https://arxiv.org/abs/2211.15089)

-   Diffusion Models: A Comprehensive Survey of Methods and Applications - [paper](https://arxiv.org/abs/2209.00796)

-   Open source diffusion implementations (contains Gaussian MNIST model) - [website](https://vinija.ai/models/diffusion/)

# Graphs

Place graphs here that help visualize the process to guide understanding and debugging.

-   SDE process Desmos plot - [website](https://www.desmos.com/calculator/rjkzmwuny0)

We have previously seen bad performance when it comes to later steps in sampling. In particular, it appears to over transform the images around 75% of the way through sampling (e.g., 750/1000 steps). Consider steps \~650 versus \~800 versus 1000 (sample on top, scores on bottom):

![](docs/class%5Cresults%5Cartifacting%5C62.png)

![](docs/class%5Cresults%5Cartifacting%5C79.png)

![](docs/class%5Cresults%5Cartifacting%5C99.png)

Note that above, the images are from different sampling runs, but the same behaviour is exhibited run to run. A full sample run is as follows:

![](docs/class%5Cresults%5Cartifacting%5Cover_transform.gif)

We can also obtain ok results if we cut off the sampling process part way (around step 700-750):

![](docs/class%5Cresults%5Cartifacting%5Ccut_off_sample.gif)

Interestingly, we also observe that if we sample from out of the training time bounds, i.e, outside of`[t_min, t_max]` we no longer have the same poor sampling behaviour as above. We are unsure of why this is the case.

![](docs/class%5Cresults%5Cout_of_bound_sampling%5C98.png)

![](docs/class%5Cresults%5Cout_of_bound_sampling%5Csample.gif)