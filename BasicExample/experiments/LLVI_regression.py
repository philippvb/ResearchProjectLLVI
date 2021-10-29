import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI')

import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

def mapping(x, mu=3,  noise=0.1):
    return 3 * torch.exp(-0.5*torch.square(x-mu)) * torch.sin(1.5*x)  + noise * torch.randn_like(x)

# def mapping(x, noise=0.1):
#     return torch.sin(x) + noise * torch.randn_like(x)

lower = 0
upper = 4

cluster_pos = [1,3]
total_points = 256
cluster_points = total_points // len(cluster_pos)
x = torch.cat([mean + torch.rand(cluster_points) for mean in cluster_pos])
x_true = torch.linspace(lower, upper, 100)
y_true = mapping(x_true, noise=False)
y = mapping(x, noise=0.01)


plt.figure()
plt.plot(x_true, y_true)
plt.scatter(x,y)
# plt.show()
# raise ValueError

class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 200)
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        # h3 = F.relu(self.fc3(h2))
        return h2

feature_extractor = FC_Net()

from BasicExample.src.LLVI_network import LLVI_network_diagonal, Loss, LLVI_network_KFac
model = LLVI_network_diagonal(feature_extractor=feature_extractor,
feature_dim=200, out_dim=1,
# A_dim=20, B_dim=10,
prior_mu=0, prior_log_var=-2,
init_ll_mu=0, init_ll_log_var=0,#init_ll_cov_scaling=0.1,
tau=0, lr=1e-4,  bias=False, loss=Loss.MSE, wdecay=0.1)

batch_size = 16
random_permutation = torch.randperm(len(x))
x_batch = torch.split(torch.unsqueeze(x[random_permutation], dim=1), batch_size)
y_batch =  torch.split(torch.unsqueeze(y[random_permutation], dim=1), batch_size)


print(model.ll_mu[:10], model.ll_log_var[:10])
# model.train_without_VI(list(zip(x_batch, y_batch)), epochs=300)
# model.train_LL(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=1000, samples=30, train_hyper=False, update_freq=10)
model.train_model(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=1000, samples=30, train_hyper=False, update_freq=10)
print(model.ll_mu[:10], model.ll_log_var[:10])

test_x = torch.unsqueeze(torch.linspace(lower, upper, 300), dim=1)
samples=100
with torch.no_grad():
    # test_y = model.forward_ML_estimate(test_x)
    y_values = model(test_x, samples=samples)[0]
    y_mean = torch.mean(y_values, dim=0)
    y_std = torch.std(y_values, dim=0)
# plt.scatter(test_x, test_y, color="red")
plt.plot(test_x, y_mean)
plt.plot(test_x, y_mean+y_std)
plt.plot(test_x, y_mean-y_std)
plt.ylim(-4,1)
plt.show()
