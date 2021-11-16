from two_moons_dataset import create_test_points, create_train_set

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
        self.fc3 = nn.Linear(20, out_dim)
        self.nll = nn.ReLU()
    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        h3 = self.fc3(h2)
        return h3

# create dataset
x, y = create_train_set(n_datapoints=256, noise=0.1)
batch_size=32
laplace_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

# init model
torch.manual_seed(0)
laplace_model = FC_Net_Laplace(out_dim=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(laplace_model.parameters(), weight_decay=5e-4, lr=1e-3)

# train
epochs = 1000
pbar = tqdm(range(epochs))
for i in pbar:
    for X_batch, y_batch in laplace_loader:
        optimizer.zero_grad()
        loss = criterion(laplace_model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {round(loss.item(), 2)}")

# define laplace
la = Laplace(laplace_model, "classification",
    subset_of_weights="last_layer", hessian_structure="full",
    prior_precision=5e-4) # prior precision is set to wdecay
la.fit(laplace_loader)


fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,6))
n_test_datapoints = 100
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
X_test, X1_test, X2_test = create_test_points(-2, 3, n_test_datapoints)

with torch.no_grad():
    lik = la(X_test, link_approx='mc', n_samples=1000)
    lik_probit = la(X_test, link_approx='probit')
    # map_conf = torch.max1imum(lik, 1 - lik).reshape(n_test_datapoints, n_test_datapoints)

map_conf = lik.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
cax1 = ax1.contourf(X1_test, X2_test, map_conf, cmap="binary")
cbar = fig.colorbar(cax1, ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax1.set_title("MC")

map_conf_probit = lik_probit.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
cax2= ax2.contourf(X1_test, X2_test, map_conf_probit, cmap="binary")
ax2.set_title("Probit")


ax1.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
ax2.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Regression/Laplace/comparison.jpg")
plt.show()



