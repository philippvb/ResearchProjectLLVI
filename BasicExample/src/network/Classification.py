import torch
from torch import nn
from src.network import LLVINetwork, LikApprox
from src.log_likelihood import LogLikelihood
from src.log_likelihood.Classification import Categorical, Categorical2Classes
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any
from src.network.feature_extractor import FC_Net
from src.weight_distribution import WeightDistribution


class LLVIClassification(LLVINetwork):
    def __init__(self, feature_dim: int, out_dim: int, feature_extractor: FC_Net, weight_dist: WeightDistribution, prior_mu: int = 0, prior_log_var: int = 0,  tau: int = 0.01, lr: int = 0.01, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        if out_dim == 1:
            loss_fun = Categorical2Classes()
        else:
            loss_fun = Categorical()
        super().__init__(feature_dim, out_dim, feature_extractor, weight_dist, loss_fun, prior_mu=prior_mu, prior_log_var=prior_log_var, tau=tau, lr=lr, optimizer_type=optimizer_type)

    def forward(self, x: torch.Tensor, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        if method == LikApprox.MONTECARLO:
            pred = self.forward_MC(x, **method_kwargs)
            lik = torch.softmax(pred, dim=-1)
            lik = torch.mean(lik, 0)
            return lik
        else:
            raise ValueError("Not implemented")
