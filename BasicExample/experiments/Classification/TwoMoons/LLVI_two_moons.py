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
from datasets.Classification.TwoMoons import create_test_points, create_train_set

n_datapoints=1024
batch_size = 32
x, y = create_train_set(n_datapoints=n_datapoints, noise=0.2)
train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

lr = 1e-3
feature_extractor = FC_Net(layers=[2, 20, 20],lr=lr, weight_decay=5e-4, optimizer=torch.optim.Adam, nll=torch.nn.LeakyReLU())

# from src.weight_distribution.Diagonal import Diagonal
from src.weight_distribution.Full import FullCovariance
# dist = Diagonal(20, 2, lr=lr)
dist = FullCovariance(20, 2, lr=lr)

net = LLVIClassification(20, 2, feature_extractor, dist,
prior_log_var=-1, optimizer_type=torch.optim.Adam,
tau=1, lr=lr)
net.train_without_VI(train_loader, epochs=10)
# test the accuracy
pred_test = torch.argmax(net(x, method=PredictApprox.MONTECARLO, samples=1000), dim=1)
print("accuracy", torch.mean((pred_test == y).float()).item())
net.train_model(train_loader, epochs=500, n_datapoints=n_datapoints, samples=5, method=LikApprox.CLOSEDFORM, approx_name="multidelta")
# net.train_ll_only(train_loader, epochs=200, n_datapoints=n_datapoints, samples=5, method=LikApprox.MONTECARLO)
# test the accuracy
pred_test = torch.argmax(net(x, method=PredictApprox.MONTECARLO, samples=1000), dim=1)
print("accuracy", torch.mean((pred_test == y).float()).item())

n_test_datapoints = 100
X_test, X1_test, X2_test = create_test_points(-2, 3, n_test_datapoints)
fig, ax = plt.subplots(figsize=(10,6))

with torch.no_grad():
    lik = net(X_test, method = PredictApprox.MONTECARLO, samples=1000)

map_conf = lik.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
cax1 = ax.contourf(X1_test, X2_test, map_conf, cmap="binary")
cbar = fig.colorbar(cax1, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
# plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Classification/init_laplace/ll_train_only.jpg")
plt.show()
