import torch
from torch.distributions import Normal
from matplotlib import pyplot as plt
import math
transform = torch.nn.LogSigmoid()

sigma = torch.tensor(1)
mean_list = torch.linspace(-10, 10, 1000)
exp_list = torch.zeros_like(mean_list)
exp_approx_list = transform(mean_list / torch.sqrt(1 + 8/math.pi * torch.square(sigma)))

for index, mean in enumerate(mean_list):
    input_dist = Normal(mean, sigma)
    input_values = input_dist.sample((20000,))
    expectation = torch.mean(transform(input_values))
    exp_list[index] = expectation


plt.figure()
plt.plot(mean_list, exp_list, label="MC")
plt.plot(mean_list, exp_approx_list, label="Approximation")
plt.legend()
plt.show()
print(expectation)

