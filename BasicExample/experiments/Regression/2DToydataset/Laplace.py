import sys
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample")

from torch import nn
import torch
from torch.utils import data
from datasets.Regression.toydataset import create_dataset_loader, sinus_mapping
from tqdm import tqdm
from laplace import Laplace
torch.manual_seed(2)

train_set, test_set = create_dataset_loader(lower=-5, upper=7, mapping=sinus_mapping,cluster_pos=[-0.5,2], data_noise=0.3, n_datapoints=256, batch_size=32)

from src.network.feature_extractor import FC_Net
# laplace_model = FC_Net([1,100,200,1], lr=1e-2)
laplace_model = torch.nn.Sequential(nn.Linear(1, 100), nn.Tanh(), nn.Linear(100, 200), nn.ReLU(), nn.Linear(200,1))
laplace_model.optimizer = torch.optim.Adam(laplace_model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()
for i in tqdm(range(300)):
    for X_batch, y_batch in train_set:
        laplace_model.optimizer.zero_grad()
        loss = criterion(laplace_model(X_batch), y_batch)
        loss.backward()
        laplace_model.optimizer.step()


la = Laplace(laplace_model, 'regression', subset_of_weights='last_layer', hessian_structure='diag')
la.model_config = {"kernel_name": "LLLaplace full hessian"}
la.fit(train_set)
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in tqdm(range(300)):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

new_model = laplace_model.delete_last_layer()

from src.log_likelihood.Regression import RegressionNoNoise
lik = RegressionNoNoise()

from src.weight_distribution.Diagonal import Diagonal
from src.weight_distribution.Full import FullCovariance
# dist = Diagonal(100, 1, lr=1e-4)
dist = FullCovariance(200, 1, lr=1e-4)
dist.update_cov(la.posterior_covariance)

from src.network.Regression import LLVIRegression

vi_net = LLVIRegression(200, 1, new_model, dist)

vi_net.train_model(train_set, 256, tau=0.1)