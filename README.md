# Simplex Diffusion

A general overview of the project can be found in the report in `docs/report.pdf`

The CSC413/2516 class report can be found along with organized results in `docs/class/`

# Simplex Diffusion Plan

## TODO for meeting
* Only 3 weeks left!
* Communication channel - (slack?)


## Baselines
* Categorical Diffusion: [paper](https://arxiv.org/abs/2107.03006)
    * [code1](https://github.com/HKUNLP/reparam-discrete-diffusion) | [code2](https://github.com/samb-t/unleashing-transformers)
* Categorical SDE w Simplex Diffusion: [paper](https://arxiv.org/abs/2210.14784) | No Code
* Discrete Flows: [paper](https://arxiv.org/abs/1905.10347) | [code?](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/discrete_flows.py)
* Analog Bits: [paper](https://arxiv.org/abs/2208.04202) | [code](https://github.com/google-research/pix2seq)

#### Viewed, but probably not
* DiffusER: [paper](https://arxiv.org/abs/2210.16886)
* Continuous Time Discrete Diffusion: [paper](https://arxiv.org/abs/2205.14987)
    * TODO: read this!!

## Experiments
* CIFAR-10 Image Generation
    * FID / Inception type scores
    * Unconditional and Conditional

* Limited-vocab Text Generation
    * See 4.4 in discrete flow, something like this...

* RL with Classifier Guidance
    * [Diffuser](https://arxiv.org/abs/2205.09991) with discrete action and state space

## Simplex Diffusion Progress
* Code forward process, test mag of score distribution
* Implimentation after should be easy.

## Thoughts
Is there a task where we can show that have points on the simplex is uniquely useful?

## Potential?
semantic segmentation thing we talked about
that could be nice

# TODO
## Image
* dataloader, neural net etc is done
* Griffin

## Text
* nothing done
* dataloader, transformer model etc
* model this off our baseline

* Eric


* also look at 'cat_*.py'

## Baseline Model
* ?? maybe cut from other models
# Simplex Diffusion Plan

## TODO for meeting
* Only 3 weeks left!
* Communication channel - (slack?)


## Baselines
* Categorical Diffusion: [paper](https://arxiv.org/abs/2107.03006)
    * [code1](https://github.com/HKUNLP/reparam-discrete-diffusion) | [code2](https://github.com/samb-t/unleashing-transformers)
* Categorical SDE w Simplex Diffusion: [paper](https://arxiv.org/abs/2210.14784) | No Code

---

#### Viewed, but probably not
* Discrete Flows: [paper](https://arxiv.org/abs/1905.10347) | [code?](https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/discrete_flows.py)
* Analog Bits: [paper](https://arxiv.org/abs/2208.04202) | [code](https://github.com/google-research/pix2seq)
* DiffusER: [paper](https://arxiv.org/abs/2210.16886)
* Continuous Time Discrete Diffusion: [paper](https://arxiv.org/abs/2205.14987)
    * TODO: read this!!

---

## Experiments
* CIFAR-10 Image Generation
    * FID / Inception type scores
    * Unconditional and Conditional

* Limited-vocab Text Generation
    * See 4.4 in discrete flow, something like this...

* RL with Classifier Guidance
    * [Diffuser](https://arxiv.org/abs/2205.09991) with discrete action and state space

## Simplex Diffusion Progress
* Code forward process, test mag of score distribution
* Implimentation after should be easy.

## Thoughts
Is there a task where we can show that have points on the simplex is uniquely useful?

## Potential?
semantic segmentation thing we talked about
that could be nice

# TODO
## Simplex Diffusion
* In both gaussian space and simplex space
* Griffin and Mihai

## Text
* nothing done
* dataloader, transformer model etc
* model this off our baseline
* Eric

## Baseline Model
* ?? maybe cut from other models
* also look at 'cat_*.py'
* Thor
