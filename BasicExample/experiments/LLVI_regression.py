import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import torch
from matplotlib import pyplot as plt

def mapping(x, noise=True):
    mu = 3
    return torch.exp(-0.5*torch.square(x-mu)) * torch.sin(2*x) + 0.1*x + noise * 0.1*torch.randn_like(x)

lower = 0
upper = 5

cluster_pos = [1,3]
x = torch.cat([mean + torch.rand(20) for mean in cluster_pos])
x_true = torch.linspace(lower, upper, 100)
y_true = mapping(x_true, noise=False)
y = mapping(x)

plt.plot(x_true, y_true)
plt.scatter(x,y)
plt.show()
