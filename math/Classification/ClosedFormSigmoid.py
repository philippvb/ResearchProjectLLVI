import torch
from torch.distributions import Normal
from matplotlib import pyplot as plt
transform = torch.nn.LogSigmoid()

sigma = torch.tensor(10)
mean_list = torch.linspace(-20, 20, 1000)
exp_list = torch.zeros_like(mean_list)
exp_2_list = transform(mean_list)#/torch.sqrt(sigma))
for index, mean in enumerate(mean_list):
    input_dist = Normal(mean, sigma)

    input_values = input_dist.sample((10000,))
    expectation = torch.mean(transform(input_values))
    exp_list[index] = expectation


plt.figure()
plt.plot(mean_list, exp_list, label="MC")
plt.plot(mean_list, exp_2_list, label="Pred")
plt.legend()
plt.show()
print(expectation)

