import sys
import torch 
from process import Process 

x = torch.tensor([4.0, 2.0, 1.5, 0.5], requires_grad=True)
out = torch.sin(x) * torch.cos(x) + x.pow(2)
# Pass tensor of ones, each for each item in x
out.backward(torch.ones_like(x))
print(x.grad)
