import torch
from torch import nn
from src.network import LLVINetwork, LikApprox, PredictApprox
from src.log_likelihood import LogLikelihood
from src.log_likelihood.Classification import Categorical, Categorical2Classes
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any
from src.network.feature_extractor import FC_Net
from src.weight_distribution import WeightDistribution
from torch.distributions import MultivariateNormal
import math


class LLVIClassification(LLVINetwork):
    def __init__(self, feature_dim: int, out_dim: int, feature_extractor: FC_Net, weight_dist: WeightDistribution, prior_mu: int = 0, prior_log_var: int = 0,  tau: int = 0.01, lr: int = 0.01, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        if out_dim == 1:
            loss_fun = Categorical2Classes()
        else:
            loss_fun = Categorical()
        super().__init__(feature_dim, out_dim, feature_extractor, weight_dist, loss_fun, prior_mu=prior_mu, prior_log_var=prior_log_var, tau=tau, lr=lr, optimizer_type=optimizer_type)

    def forward(self, x: torch.Tensor, method:PredictApprox=PredictApprox.MONTECARLO, **method_kwargs) -> torch.Tensor:
        if self.out_dim == 1:
            pred_mean, pred_cov = self.forward_single(x) # single output forward pass
        else:
            pred_mean, pred_cov = self.forward_multi(x) # multi output

        if method == PredictApprox.MONTECARLO:
            samples = MultivariateNormal(pred_mean, pred_cov).sample((method_kwargs["samples"], ))
            if self.out_dim == 1:
                return torch.sigmoid(samples, dim=-1).mean(dim=0)
            else:
                return torch.softmax(samples, dim=-1).mean(dim=0)

        elif method == PredictApprox.PROBIT:
            if self.out_dim == 1:
                return self.probit_approx(pred_mean, torch.diag(pred_cov))
            else:
                pred_var = pred_cov.diagonal(dim1=-2, dim2=-1)
                return torch.softmax(self.probit_approx(pred_mean, pred_var), dim=-1)
        else:
            raise NotImplementedError(f"Approximation method {method} not implemented")

    def probit_approx(self, mean, cov):
        """Probit approximation given a mean and a covariance
        """
        return mean / torch.sqrt(1 + math.pi * cov / 8)


