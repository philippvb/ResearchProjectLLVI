import math
import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample')

from datasets.Classification.TwoMoons import create_test_points, create_train_set
from src.network.feature_extractor import FC_Net
from torch import _test_serialization_subcmul, nn, optim
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib


class CNN(nn.Module):
    def __init__(self, optimizer=optim.Adam, **optim_kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 50)
        self.fc3 = nn.Linear(50, 10, bias=False)
        self.optimizer: optim = optimizer(self.parameters(), **optim_kwargs)
        self.weight_decay = optim_kwargs["weight_decay"]

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x #F.softmax(x, dim=-1)


class VICNN(nn.Module):
    def __init__(self, optimizer=optim.Adam, **optim_kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 50)
        self.optimizer: optim = optimizer(self.parameters(), **optim_kwargs)
        self.weight_decay = optim_kwargs["weight_decay"]

    def forward(self, x):
        x = self.pool(F.sigmoid(self.conv1(x)))
        x = self.pool(F.sigmoid(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x




import torchvision
batch_size_train = 32
batch_size_test = 60000
filepath = "/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/datasets/Classification"
# create dataset
dataset = torchvision.datasets.MNIST(filepath, train=True, download=False,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_test, shuffle=True)
n_datapoints = batch_size_train * len(train_loader)

# init model
torch.manual_seed(3)
weight_decay = 5e-4
laplace_model = CNN(weight_decay=weight_decay)
criterion = torch.nn.CrossEntropyLoss()

# train
epochs = 1
pbar = tqdm(range(epochs))
for i in pbar:
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        laplace_model.optimizer.zero_grad()
        output = laplace_model(X_batch)
        output = F.log_softmax(output, dim=-1)
        # output = torch.log(output)
        loss = criterion(output, y_batch)
        loss.backward()
        epoch_loss += loss.item()
        laplace_model.optimizer.step()

    pbar.set_description(f"Loss: {round(float(torch.mean(loss)), 2)}")

# define laplace
la = Laplace(laplace_model, "classification",
    subset_of_weights="last_layer", hessian_structure="diag",
    prior_precision=5e-4) # prior precision is set to wdecay
la.fit(train_loader)

for X_batch, y_batch in test_loader:
    predictions = la(X_batch, link_approx='mc', n_samples=1000)
    pred_test = torch.argmax(predictions, dim=1)
    print("Accuracy with Laplace", torch.mean((pred_test == y_batch).float()).item())




# define weight distribution and update values
from src.weight_distribution.Full import FullCovariance
from src.weight_distribution.Diagonal import Diagonal
dist = FullCovariance(50, 10, lr=1e-5)

dist = Diagonal(50, 10, lr=1e-4)
dist.update_var(torch.reshape(la.posterior_variance, (50, 10)))
dist.update_mean(torch.t(laplace_model.fc3.weight))


vi_model = VICNN(weight_decay=weight_decay)
with torch.no_grad():
    vi_model.load_state_dict(laplace_model.state_dict(), strict=False)


# define VI model
from src.network.Classification import LLVIClassification
from src.network import PredictApprox, LikApprox
prior_log_var = math.log(1/(weight_decay * n_datapoints))
net = LLVIClassification(50, 10, vi_model, dist, prior_log_var=prior_log_var, optimizer_type=torch.optim.Adam,
tau=1, lr=1e-5)

for X_batch, y_batch in test_loader:
    predictions = net(X_batch, method=PredictApprox.MONTECARLO, samples=100)
    pred_test = torch.argmax(predictions, dim=1)
    print("Accuracy with VI before Training", torch.mean((pred_test == y_batch).float()).item())

# net.train_hyper(laplace_loader, epochs=500, samples=1000)
# net.train_model(train_loader, epochs=5, n_datapoints=n_datapoints, method=LikApprox.MONTECARLO, samples=10)
# net.train_em_style(laplace_loader, n_datapoints, total_epochs=100, inner_epochs_fe=1, inner_epochs_vi=5, method=LikApprox.MONTECARLO, samples=10)
net.train_model(train_loader, epochs=10, n_datapoints=n_datapoints, samples=10, method=LikApprox.MONTECARLO, approx_name="jennsen")

for X_batch, y_batch in test_loader:
    predictions = net(X_batch, method=PredictApprox.MONTECARLO, samples=100)
    pred_test = torch.argmax(predictions, dim=1)
    print("Accuracy with VI after Training", torch.mean((pred_test == y_batch).float()).item())