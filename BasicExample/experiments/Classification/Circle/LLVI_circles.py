import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample')

from sklearn import datasets
import numpy as np
import torch
torch.manual_seed(1)
from torch import nn, optim
import torch.nn.functional as F 
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
import matplotlib
from src.network.feature_extractor import FC_Net
from src.network.Classification import LLVIClassification
from src.network import LikApprox, PredictApprox
from datasets.Classification.circles import create_train_set, create_test_points

n_datapoints=256
batch_size = 32
x, y = create_train_set(n_datapoints=n_datapoints, noise=0.2)
train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

lr = 1e-3
feature_extractor = FC_Net(layers=[2, 20, 20],lr=lr, weight_decay=5e-4, optimizer=torch.optim.Adam, nll=torch.nn.LeakyReLU())

from src.weight_distribution.Diagonal import Diagonal
from src.weight_distribution.Full import FullCovariance
dist = Diagonal(20, 2, lr=lr)
# dist = FullCovariance(20, 2, lr=lr)

net = LLVIClassification(20, 2, feature_extractor, dist,
prior_log_var=-7, optimizer_type=torch.optim.Adam,
tau=1, lr=lr)
# net.train_without_VI(train_loader, epochs=100)
# test the accuracy
# pred_test = torch.argmax(net(x, method=PredictApprox.MAXIMUMLIKELIHOOD, samples=1000), dim=1)
# print("accuracy", torch.mean((pred_test == y).float()).item())
# net.train_model(train_loader, epochs=200, n_datapoints=n_datapoints, samples=10, method=LikApprox.MONTECARLO)
net.train_em_style(train_loader, n_datapoints, total_epochs=500, inner_epochs_fe=1, inner_epochs_vi=3, method=LikApprox.MONTECARLO, samples=10)
# net.train_ll_only(train_loader, epochs=200, n_datapoints=n_datapoints, samples=5, method=LikApprox.MONTECARLO)
# test the accuracy
pred_test = torch.argmax(net(x, method=PredictApprox.MONTECARLO, samples=1000), dim=1)
# pred_test = torch.argmax(net.forward_ML(x), dim=-1)
print("accuracy", torch.mean((pred_test == y).float()).item())

n_test_datapoints = 100
X_test, X1_test, X2_test = create_test_points(-3, 3, n_test_datapoints)
fig, ax = plt.subplots(figsize=(10,6))
cbar_ax = fig.add_axes([1, 0.15, 0.05, 0.7])

with torch.no_grad():
    lik = net(X_test, method = PredictApprox.MONTECARLO, samples=1000)
    # lik = torch.softmax(net.forward_ML(X_test), dim=-1)

pred = torch.argmax(lik, dim=1)
lik[:,1] *=-1
map_conf = lik[torch.arange(lik.shape[0]), pred].reshape(n_test_datapoints, n_test_datapoints)
cax1 = ax.contourf(X1_test, X2_test, map_conf, cmap="bwr", alpha=0.5, levels=10)
cbar = fig.colorbar(cax1, cbar_ax, ticks=[-1, -0.5, 0, 0.5, 1])
ax.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
# plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Classification/Circles/prediction.jpg")
plt.show()
