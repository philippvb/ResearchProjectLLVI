import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample')

from two_moons_dataset import create_test_points, create_train_set
from src.network.feature_extractor import FC_Net
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib

class FC_Net_Laplace(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, out_dim, bias=False)
        self.nll = nn.LeakyReLU() # dont use normal relu here since otherwise some values can be 0 later
    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3

    def delete_last_layer(self):
        self.forward = self.new_forward

    def new_forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        return h2



# create dataset
n_datapoints = 256
x, y = create_train_set(n_datapoints=n_datapoints, noise=0.2)
batch_size=32
laplace_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)
n_test_datapoints = 70
X_test, X1_test, X2_test = create_test_points(-2, 3, n_test_datapoints)

# init model
torch.manual_seed(3)
laplace_model = FC_Net_Laplace(2)
criterion = torch.nn.CrossEntropyLoss()
laplace_model.optimizer = torch.optim.Adam(laplace_model.parameters(), weight_decay=5e-4, lr=1e-3)

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
la = Laplace(laplace_model, "classification",
    subset_of_weights="last_layer", hessian_structure="full",
    prior_precision=5e-4) # prior precision is set to wdecay
la.fit(laplace_loader)

# init figure
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))

# plot laplace predictions
with torch.no_grad():
    lik = la(X_test, link_approx='mc', n_samples=1000)
map_conf = lik.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
cax1 = ax1.contourf(X1_test, X2_test, map_conf, cmap="binary")
cbar = fig.colorbar(cax1, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_title("Laplace pretaining")
ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)


# define weight distribution and update values
from src.weight_distribution.Full import FullCovariance
dist = FullCovariance(20, 2, lr=1e-4)
dist.update_cov(la.posterior_covariance)
dist.update_mean(torch.t(laplace_model.fc3.weight))

# define weight distribution and update values
# from src.weight_distribution.Diagonal import Diagonal
# dist = Diagonal(20, 2, lr=1e-4)
# dist.update_var(torch.reshape(torch.diag(la.posterior_covariance), (20, 2)))
# dist.update_mean(torch.t(laplace_model.fc3.weight))

# delete last layer
laplace_model.delete_last_layer()


# define VI model
from src.network.Classification import LLVIClassification
from src.network import PredictApprox, LikApprox
net = LLVIClassification(20, 2, laplace_model, dist, prior_log_var=1,
tau=0.1, lr=1e-3)

# net.prior_mu = nn.Parameter(torch.t(laplace_model.fc3.weight).detach().clone(), requires_grad=True)
# net.prior_log_var = nn.Parameter(torch.reshape(torch.log(torch.diag(la.posterior_covariance)), net.prior_mu.shape).detach().clone(), requires_grad=True)

# net.train_without_VI(laplace_loader, epochs=1000)
# print(torch.eq(la.posterior_covariance, dist.get_cov()))
# net.train_model(laplace_loader, epochs=1000, n_datapoints=n_datapoints, samples=5000, method=LikApprox.MONTECARLO)#, train_hyper=True, update_freq=5)
# print(net.feature_extractor.state_dict())

# test the accuracy
pred_test = torch.argmax(net(x, method=PredictApprox.MONTECARLO, samples=1000), dim=1)
print("accuracy", torch.mean((pred_test == y).float()).item())

with torch.no_grad():
    lik2 = net(X_test, method=PredictApprox.MONTECARLO, samples=10000)

map_conf = lik2.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
cax2 = ax2.contourf(X1_test, X2_test, map_conf, cmap="binary")
ax2.set_title("VI after Laplace")
ax2.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)

plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Classification/init_laplace/laplace_as_prior_2")
plt.show()
