import math
import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample')

from datasets.Regression.toydataset import create_dataset, sinus_mapping, dataset_to_loader, visualize_predictions
from src.network.feature_extractor import FC_Net
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

class FC_Net_Laplace(nn.Module):
    def __init__(self, out_dim, weight_decay):
        super().__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, out_dim, bias=False)
        self.nll = nn.Tanh() # dont use normal relu here since otherwise some values can be 0 later
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=weight_decay, lr=1e-4)

    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3

    def delete_last_layer(self):
        self.forward = self.new_forward
        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=0.01, lr=1e-4)

    def new_forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        return h2

n_datapoints=256
data_noise = 0.2
x_train, y_train, x_test, y_test = create_dataset(lower=-5, upper=7, mapping=sinus_mapping,cluster_pos=[-0.5,2], data_noise=data_noise, n_datapoints=n_datapoints)

# create dataset
batch_size = 16
laplace_loader = DataLoader(TensorDataset(torch.unsqueeze(x_train, dim=1), torch.unsqueeze(y_train, dim=1)), batch_size=batch_size)


# init model
torch.manual_seed(3)
weight_decay = 0.01
laplace_model = FC_Net_Laplace(1, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()

# train
epochs = 100
pbar = tqdm(range(epochs))
for i in pbar:
    for X_batch, y_batch in laplace_loader:
        laplace_model.optimizer.zero_grad()
        loss = criterion(laplace_model(X_batch), y_batch)
        loss.backward()
        laplace_model.optimizer.step()
        pbar.set_description(f"Loss: {round(loss.item(), 2)}")

# define laplace
la = Laplace(laplace_model, "regression",
    subset_of_weights="last_layer", hessian_structure="diag")
la.fit(laplace_loader)

log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in tqdm(range(100)):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

# init figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
ax1.set_title("Laplace pretraining")
ax2.set_title("VI with Laplace init, Monte Carlo")
visualize_predictions(la, ax1, x_train, y_train, x_test, y_test, data_noise=data_noise, laplace=True)

plt.show()
raise ValueError
# define VI model
from src.network.feature_extractor import FC_Net
from src.network.Regression import LLVIRegression
from src.network import PredictApprox, LikApprox


lr = 1e-4
feature_extractor = FC_Net(layers=[1, 200, 100], nll = torch.nn.Tanh(),lr=lr, weight_decay=0.1)

from src.weight_distribution.Full import FullCovariance
dist = FullCovariance(100, 1, lr=lr, init_log_var=-0.5)

# updating the distribution
dist.update_cov(la.posterior_covariance)
dist.update_mean(torch.t(laplace_model.fc3.weight))
# delete last layer
laplace_model.delete_last_layer()

prior_log_var = math.log(1/(weight_decay * n_datapoints))
net = LLVIRegression(100, 1, laplace_model, dist, prior_log_var=prior_log_var,
tau=1, data_log_var=torch.log(torch.square(la.sigma_noise.detach().clone())),
 lr=lr)
# net.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# net.train_model(list(zip(x_batch, y_batch)), epochs=500, n_datapoints=512, samples=1, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)

# net.train_without_VI(laplace_loader, epochs=100)
net.train_model(laplace_loader, epochs=100, n_datapoints=256, samples=100, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=1)

visualize_predictions(net, ax2, x_train, y_train, x_test, y_test, data_noise=data_noise)
# plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Regression/Laplace/laplace_init_mc.jpg")
plt.show()