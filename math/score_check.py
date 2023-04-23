import os, sys
import torch
import numpy as np

'''
checking properties of generalized logit-gauss
    - is the score correct?
    - check via finite differences
'''

# TODO: impliment the log pdf directly for numerical stability
# see here for the math: https://hackmd.io/@PuY5GAnCRg68x1VJTx1ZVA/HyIKl2ZXh 