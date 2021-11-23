from src.weight_distribution import WeightDistribution
import torch
from torch import nn, optim

class Diagonal(WeightDistribution):
    def __init__(self, feature_dim:int, out_dim:int, init_mu=0, init_log_var=0, optimizer=optim.SGD, **optim_kwargs) -> None:
        super().__init__(feature_dim, out_dim)
        self.mu = nn.Parameter(init_mu + torch.randn(feature_dim, out_dim), requires_grad=True)
        self.log_var = nn.Parameter(init_log_var + torch.randn_like(self.mu), requires_grad=True)
        self.optimizer = optimizer([self.mu, self.log_var], **optim_kwargs)
        self.n_parameters = feature_dim * out_dim

    def get_mu(self) -> torch.Tensor:
        return self.mu

    def get_cov(self) -> torch.Tensor:
        return torch.diag(torch.flatten(torch.exp(self.log_var)))

    def sample(self, samples=10):
        std = torch.multiply(torch.exp(0.5 * self.log_var),  torch.randn((samples, ) + self.log_var.size()))
        return self.mu + std

    def KL_div(self, prior_mu, prior_log_var) -> torch.Tensor:
        div = 0.5 * (torch.sum(prior_log_var) - torch.sum(self.log_var) - self.n_parameters + torch.sum(torch.exp(self.log_var - prior_log_var)) + torch.sum(torch.div(torch.square(prior_mu - self.mu), torch.exp(prior_log_var))))
        return div

    def update_var(self, new_var:torch.Tensor) -> None:
        new_var = new_var.detach().clone() # make sure we create new matrix
        assert new_var.shape == self.log_var.shape
        self.log_var = nn.Parameter(torch.log(new_var), requires_grad=True)
        self.optimizer.param_groups[0]["params"] = [self.mu, self.log_var] # update parameters

    def update_mean(self, new_mean: torch.Tensor) -> None:
        new_mean = new_mean.detach().clone()
        assert new_mean.shape == self.mu.shape
        self.mu = nn.Parameter(new_mean, requires_grad=True)
        self.optimizer.param_groups[0]["params"] = [self.mu, self.log_var] # update parameters
