import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI')

import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

def mapping(x, mu=3,  noise=0.1):
    # return 10 * torch.exp(-0.5*torch.square(x-mu)) * torch.sin(0.3* x + 4)  + noise * torch.randn_like(x)
    return nn.Sigmoid()(x)

# def mapping(x, noise=0.1):
#     return torch.sin(x) + noise * torch.randn_like(x)

lower = 0
upper = 6

cluster_pos = [1,5]
total_points = 256
cluster_points = total_points // len(cluster_pos)
x = torch.cat([mean + 0.5 * torch.rand(cluster_points) for mean in cluster_pos])
x_true = torch.linspace(lower, upper, 100)
y_true = mapping(x_true, noise=False)
y = mapping(x, noise=0.2)


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
        self.nll = nn.LeakyReLU()
    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        # h3 = F.relu(self.fc3(h2))
        return h2

feature_extractor = FC_Net()

from BasicExample.src.LLVI_network import LLVI_network_diagonal, Log_likelihood_type, LLVI_network_KFac, LLVI_network_full_Cov
# model = LLVI_network_diagonal(feature_extractor=feature_extractor,
# feature_dim=200, out_dim=1,
# # A_dim=20, B_dim=10,
# prior_mu=0, prior_log_var=-6,
# init_ll_mu=0, init_ll_log_var=-2,#init_ll_cov_scaling=0.1,
# tau=0.01, lr=1e-4,  bias=False, loss=Log_likelihood_type.MSE, wdecay=0.1)

model = LLVI_network_full_Cov(feature_extractor=feature_extractor,
feature_dim=200, out_dim=1,
prior_mu=0, prior_log_var=-5,
init_ll_mu=0, init_ll_log_var=-1, init_ll_cov_scaling=0.1,
tau=0.01, lr=1e-4,  bias=False, loss=Log_likelihood_type.MSE, wdecay=0.1, data_log_var=-1)


batch_size = 16
random_permutation = torch.randperm(len(x))
x_batch = torch.split(torch.unsqueeze(x[random_permutation], dim=1), batch_size)
y_batch =  torch.split(torch.unsqueeze(y[random_permutation], dim=1), batch_size)


model.train_without_VI(list(zip(x_batch, y_batch)), epochs=10)
# model.train_LL(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=300, samples=20, train_hyper=False, update_freq=10)
model.train_model(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=1000, samples=1, train_hyper=True, update_freq=2)
print("The estimate std deviation of the data is", torch.exp(0.5 * model.data_log_var).item())


test_x = torch.unsqueeze(torch.linspace(lower-2, upper+5, 300), dim=1)

samples=100
with torch.no_grad():
    y_values = model(test_x, samples=samples)[0]
    y_mean = torch.mean(y_values, dim=0)
    y_std = torch.std(y_values, dim=0)
    plt.plot(test_x, y_mean, color="red")
    plt.plot(test_x, y_mean+1.96*y_std, color="red")
    plt.plot(test_x, y_mean-1.96*y_std, color="red")


y_mean, y_std = model.predict(test_x)
y_std = torch.sqrt(torch.diagonal(y_std))
print(y_std)
plt.plot(test_x, y_mean, color="green")
plt.plot(test_x, y_mean+1.96*y_std, color="green")
plt.plot(test_x, y_mean-1.96*y_std, color="green")
plt.ylim(torch.min(y_true) - 2, torch.max(y_true) + 2)
plt.show()
