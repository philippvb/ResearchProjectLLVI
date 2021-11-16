import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI')

import torch
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
torch.manual_seed(5)
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

def mapping(x, noise=0.1):
    return torch.sin(2.5*x) + noise * torch.randn_like(x)


lower = -5
upper = 7

cluster_pos = [-1,2]
total_points = 256
cluster_points = total_points // len(cluster_pos)
x = torch.cat([mean + torch.rand(cluster_points) for mean in cluster_pos])
x_true = torch.linspace(lower, upper, 100)
data_noise = 0.2
y_true = mapping(x_true, noise=False)
y = mapping(x, noise=data_noise)


plt.figure()
plt.plot(x_true, y_true, color="black", label="True function")
plt.scatter(x,y, s=3, color="black")
# plt.show()
# raise ValueError

class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 100)
        self.nll = nn.Tanh()
    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        return h2

feature_extractor = FC_Net()

from BasicExample.src.LLVI_network import LLVI_network_diagonal, Log_likelihood_type, LLVI_network_KFac, LLVI_network_full_Cov
model = LLVI_network_KFac(feature_extractor=feature_extractor,
feature_dim=100, out_dim=1,
A_dim=10, B_dim=10,
prior_mu=0, prior_log_var=-6,
init_ll_mu=0, init_ll_cov_scaling=0.1,
tau=0.01, lr=1e-5,  bias=False, loss=Log_likelihood_type.MSE, wdecay=0.1, data_log_var = -1)

# model = LLVI_network_full_Cov(feature_extractor=feature_extractor,
# feature_dim=100, out_dim=1,
# prior_mu=0, prior_log_var=-4,
# init_ll_mu=0, init_ll_log_var=-0.5, init_ll_cov_scaling=0.1,
# tau=0.01, lr=5e-5,  bias=False, loss=Log_likelihood_type.MSE, wdecay=0.1, data_log_var=-1)


batch_size = 16
random_permutation = torch.randperm(len(x))
x_batch = torch.split(torch.unsqueeze(x[random_permutation], dim=1), batch_size)
y_batch =  torch.split(torch.unsqueeze(y[random_permutation], dim=1), batch_size)


# model.load_ml_estimate("P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/models/Regression/Pretrained/Baseline_wdecay_e-1")
model.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# filepath = model.save_ml_estimate("P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/models/Regression/Pretrained")
# model.update_prior_mu(model.ll_mu)
# model.train_LL(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=500, samples=1, train_hyper=True, update_freq=10)
# model.update_prior_mu(model.ll_mu)
model.train_model(list(zip(x_batch, y_batch)), n_datapoints=total_points, epochs=500, samples=1, train_hyper=True, update_freq=5)
# print("The estimate std deviation of the data is", torch.exp(0.5 * model.data_log_var).item())
# filepath = "P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/models/Regression/FullCov"
# filepath = model.save(filepath)


test_x = torch.unsqueeze(torch.linspace(lower, upper, 300), dim=1)
bayesian = True
noise = True

if not bayesian:
    with torch.no_grad():
        y_mean = model.forward_ML_estimate(test_x)

else:
    with torch.no_grad():
        y_mean, y_var = model.predict_regression(test_x, noise=noise)
    y_std = torch.sqrt(torch.diagonal(y_var)) - noise * (torch.exp(0.5 * model.data_log_var).item() + data_noise)
    plt.plot(test_x, y_mean+1.96*y_std, color="orange", label="$+-1.96 \cdot \sigma$")
    plt.plot(test_x, y_mean-1.96*y_std, color="orange")
    plt.fill_between(torch.squeeze(test_x),y_mean+1.96*y_std,y_mean-1.96*y_std, alpha=0.1, color="orange")

plt.plot(test_x, y_mean, color="royalblue", label="ML prediction/Mean")
plt.ylim(torch.min(y_true) - 2, torch.max(y_true) + 2)
plt.title("Baseline")
plt.ylabel("y")
noise_txt = f"Estimated data noise: {round(torch.exp(0.5 * model.data_log_var).item(), 2)}, true noise:{data_noise}"
plt.xlabel(r'\begin{center}x\\*\textit{\small{' + noise_txt + r'}}\end{center}')
plt.legend()
plt.tight_layout()
# plt.savefig(filepath + "/result.jpg")
plt.show()
