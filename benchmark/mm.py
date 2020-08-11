import time
import torch

a = torch.ones(1000, 1000, requires_grad=True)
b = torch.ones(1000, 1000, requires_grad=True)

t1 = time.time()
c = torch.mm(a, b)
t2 = time.time()
print(t2 - t1)

d = c.sum()
print(d)
