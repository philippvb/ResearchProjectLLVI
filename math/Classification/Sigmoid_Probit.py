import torch
from torch.distributions import Normal, LogNormal
from matplotlib import pyplot as plt
import math
log_sig = torch.nn.LogSigmoid()
transform = torch.nn.Sigmoid()
# id = lambda x: x
# relu = torch.nn.LeakyReLU()
# nonlinear = torch.log
# eta = lambda x: 2* (x + torch.log(1+torch.exp(-x)))/(torch.log(torch.sqrt(torch.tensor([2 * math.pi]))) * torch.square(x))

d = Normal(0, 1)

x = torch.linspace(-3, 3, 500)
dist = Normal(0, 1)
# y_true = -nonlinear(transform(x))
y_true = -log_sig(x)
y_approx_cdf = -dist.cdf(x * 1 - 0.3)+1
y_approx_exp = torch.exp(-0.5 * x)
y_approx_square = torch.square(x)#-torch.log(1 + torch.square(x))
plt.figure()
plt.plot(x, y_true, label="y_true")
plt.plot(x, y_approx_cdf, label="cdf")
plt.plot(x, y_approx_exp, label="exp")
plt.plot(x, y_approx_square, label="square")
# plt.ylim(0, 10)
plt.legend()
plt.show()

# print(nonlinear(dist.cdf(torch.sqrt(torch.tensor([math.pi/8])) *-8)))
