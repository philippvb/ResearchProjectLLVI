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
        self.fc3 = nn.Linear(20, 40)
        self.fc_out = nn.Linear(40, out_dim)
        self.nll = nn.ReLU()
    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        h3 = self.nll(self.fc3(h2))
        return self.fc_out(h3)

# create dataset
x, y = create_train_set(n_datapoints=1024)
batch_size=32
laplace_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size)

# init model
torch.manual_seed(2)
laplace_model = FC_Net_Laplace(out_dim=2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(laplace_model.parameters(), weight_decay=0.001, lr=1e-3)

# train
epochs = 500
pbar = tqdm(range(epochs))
for i in pbar:
    for X_batch, y_batch in laplace_loader:
        optimizer.zero_grad()
        loss = criterion(laplace_model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Loss: {round(loss.item(), 2)}")

la = Laplace(laplace_model, "classification", subset_of_weights="all", hessian_structure="full")
la.fit(laplace_loader)
log_prior= torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior], lr=1e-1)
for i in tqdm(range(300)):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

fig, ax = plt.subplots()
n_test_datapoints = 70
ax.set_xlabel('x1')
ax.set_ylabel('x2')
X_test, X1_test, X2_test = create_test_points(-2, 5, n_test_datapoints)

with torch.no_grad():
    lik = la(X_test, link_approx='probit')
    # map_conf = torch.maximum(lik, 1 - lik).reshape(n_test_datapoints, n_test_datapoints)
    map_conf = lik.max(1).values.reshape(n_test_datapoints, n_test_datapoints)
    cmap = ax.contourf(X1_test, X2_test, map_conf)


ax.scatter(x[:, 0], x[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
plt.show()

