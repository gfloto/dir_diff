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
