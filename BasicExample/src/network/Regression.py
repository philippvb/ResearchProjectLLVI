import torch
from torch import nn

from src.log_likelihood import LogLikelihood
from src.log_likelihood.Regression import (ClosedFormRegression, Regression, RegressionNoNoise)

from src.network import LikApprox, LLVINetwork
from src.network.feature_extractor import FC_Net

from src.weight_distribution import WeightDistribution


class LLVIRegression(LLVINetwork):
    def __init__(self, feature_dim: int, out_dim: int, feature_extractor: FC_Net, weight_dist: WeightDistribution, prior_mu: int = 0, prior_log_var: int = 0, data_log_var: int = 0, tau: int = 0.01, lr: int = 0.01, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        loss_fun = Regression()
        self.loss_fun_ml = RegressionNoNoise()
        self.loss_fun_closed_form = ClosedFormRegression() # for closed form
        super().__init__(feature_dim, out_dim, feature_extractor, weight_dist, loss_fun, prior_mu=prior_mu, prior_log_var=prior_log_var, tau=tau, lr=lr, optimizer_type=optimizer_type)

        # the log variance of the data
        self.data_log_var = nn.Parameter(torch.tensor([data_log_var], dtype=torch.float32), requires_grad=True)
        # reinit the hyperparameter optimizer
        self.prior_optimizer: torch.optim.Optimizer = self.init_hyperparam_optimizer_with_data_log_var(optimizer_type)

    def init_hyperparam_optimizer_with_data_log_var(self, optimizer_type: torch.optim.Optimizer):
        return optimizer_type([self.prior_mu, self.prior_log_var, self.data_log_var], self.lr) # also add the data_log_var

    def compute_prediction_loss(self, data: torch.Tensor, target: torch.Tensor, method: LikApprox, **method_kwargs) -> torch.Tensor:
        if method == LikApprox.CLOSEDFORM:
            pred_mean, pred_cov = self.forward(data)
            return self.loss_fun_closed_form(pred_mean, pred_cov, target, torch.exp(self.data_log_var))
        elif method == LikApprox.MONTECARLO:
            prediction = self.forward_MC(data)
            return self.loss_fun(prediction, target, data_var = torch.exp(self.data_log_var), average=True)
        elif method == LikApprox.MAXIMUMLIKELIHOOD:
            prediction = self.forward_ML(data)
            return self.loss_fun_ml(prediction, target, average=False)
        else:
            raise ValueError(f"Method {method} not implemented")

    def predict(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mean, pred_cov = self.forward(x)
        return pred_mean, pred_cov + torch.exp(self.data_log_var)

class LLVIRegressionNoNoise(LLVINetwork):
    def __init__(self, feature_dim: int, out_dim: int, feature_extractor: FC_Net, weight_dist: WeightDistribution, loss_fun: LogLikelihood, prior_mu: int = 0, prior_log_var: int = 0, tau: int = 0.01, lr: int = 0.01, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        loss_fun = RegressionNoNoise()
        super().__init__(feature_dim, out_dim, feature_extractor, weight_dist, loss_fun, prior_mu=prior_mu, prior_log_var=prior_log_var, tau=tau, lr=lr, optimizer_type=optimizer_type)

    def predict(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pred_mean, pred_cov = self.forward(x)
        return pred_mean, pred_cov
