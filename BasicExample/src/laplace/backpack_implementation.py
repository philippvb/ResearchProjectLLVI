from enum import Enum
from turtle import forward
from typing import List

import torch
from torch import nn
from src.log_likelihood import LogLikelihood
from src.network.feature_extractor import FC_Net
from src.utils.TrainWrapper import Trainwrapper
from src.derivatives import nth_derivative
from src.log_likelihood.Regression import RegressionNoNoise
from backpack import backpack, extend
from backpack.extensions import DiagHessian, DiagGGNExact
from torch.autograd import grad

class LikApprox(Enum):
    """Available Likelihood approximations
    """
    MAXIMUMLIKELIHOOD = "ML"
    MONTECARLO = "MC"
    CLOSEDFORM = "CF"


    

class LaplaceVI(nn.Module):
    """Base class for Last-Layer Variational Inference Networks (LLVI).
    """


    def __init__(self, feature_dim: int, out_dim: int, feature_extractor:FC_Net, loss_fun:LogLikelihood, prior_mu:int=0, prior_log_var:int=0, tau:int=1e-2, lr: int = 1e-2, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        super().__init__()
        self.loss_fun = torch.nn.MSELoss(reduction="sum")
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.feature_extractor = feature_extractor
        self.tau = tau
        self.lr = lr
        self.prior_mu = prior_mu
        self.prior_log_var = torch.tensor([prior_log_var])
        # self.prior_mu: torch.Tensor = self.create_full_tensor(feature_dim, out_dim, prior_mu)
        # self.prior_log_var: torch.Tensor = self.create_full_tensor(feature_dim, out_dim, prior_log_var)
        # self.prior_optimizer: torch.optim.Optimizer = self.init_hyperparam_optimizer(optimizer_type)

        # self.ll_mu = nn.Parameter(torch.randn(feature_dim, out_dim, requires_grad=True))
        # self.ll_optimizer = optimizer_type([self.ll_mu], lr)

        self.nn_optimizers: List[torch.optim.Optimizer] = [feature_extractor.optimizer] # optimizers of the neural network

        extend(self.feature_extractor)
        extend(self)

    # def create_full_tensor(self, dim1, dim2, fill_value):
    #     return nn.Parameter(torch.full((dim1, dim2), fill_value=fill_value, requires_grad=True, dtype=torch.float32))



    def compute_ELBO(self, data:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        neg_log_joint = self.compute_neg_log_joint(data, target)
        # compute the hessian wrt log_joint but don't populate the grad field
        with backpack(DiagGGNExact()):
            grads = grad(neg_log_joint, self.feature_extractor.parameters(), create_graph=True)[0]
        hess = list(self.feature_extractor.children())[-1].weight.diag_ggn_exact
        log_det = - 1000*torch.log(hess).sum()/512 # log determinant for diagonal matrices
        loss = neg_log_joint - log_det/2  #TODO: add scaling
        return loss

    def compute_neg_log_joint(self, data, target):
        prediction = self.feature_extractor(data)
        log_lik = self.loss_fun(prediction, target)
        log_prior = self.get_log_prior(self.feature_extractor[-1].weight)
        return log_lik - log_prior

    def get_log_prior(self, weights:torch.Tensor):
        # prior_distribution = torch.distributions.MultivariateNormal(torch.full((self.feature_dim * self.out_dim, 1), fill_value=self.prior_mu), torch.exp(self.prior_log_var) * torch.eye(self.feature_dim * self.out_dim))
        # return prior_distribution.log_prob(weights.flatten())
        # just 0 1 prior
        return -weights.square().sum()/torch.exp(self.prior_log_var)

    def clear_nn_optimizers(self):
        [optimizer.zero_grad() for optimizer in self.nn_optimizers]

    def step_nn_optimizers(self):
        [optimizer.step() for optimizer in self.nn_optimizers]


    def train_without_VI(self, train_loader, epochs=1, callback=None):
        self.train()
        train_wrapper = Trainwrapper(["loss"])
        def train_step_fun(data, target):
            # clear gradients
            self.clear_nn_optimizers()
            # compute loss functions
            loss = self.compute_neg_log_joint(data, target)
            # backward pass
            loss.backward()
            self.step_nn_optimizers()
            return [loss]

        # hyper_fun = lambda train_loader: self.train_hyper_epoch(train_loader, n_datapoints, method, **method_kwargs) if train_hyper else None
        hyper_fun=None

        # return train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=1, callback=callback)
        output = train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=1, callback=callback)
        # now set cov
        self.set_cov(train_loader)
        return output
        # train functions
    def train_model(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, callback=None, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
        self.train()
        train_wrapper = Trainwrapper(["loss"])
        def train_step_fun(data, target):
            # clear gradients
            self.clear_nn_optimizers()
            # compute loss functions
            loss = self.compute_ELBO(data, target)
            # backward pass
            loss.backward()
            self.step_nn_optimizers()
            return [loss]

        # hyper_fun = lambda train_loader: self.train_hyper_epoch(train_loader, n_datapoints, method, **method_kwargs) if train_hyper else None
        hyper_fun=None

        output = train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=update_freq, callback=callback)
        # now set cov
        self.set_cov(train_loader)
        return output

    def set_cov(self, dataloader):
        self.cov_approx = torch.zeros_like(self.feature_extractor[-1].weight)
        for data, target in dataloader:
            with backpack(DiagGGNExact()):
                self.compute_neg_log_joint(data, target).backward()
        self.cov_approx = self.feature_extractor[-1].weight.diag_ggn_exact.detach()
        self.cov_approx = 1/self.cov_approx
        self.cov_approx/=512
        self.cov_approx = torch.diag(self.cov_approx.flatten())



    def forward(self, data):
        removed = list(self.feature_extractor.children())[:-1]
        self.ll_mu = list(self.feature_extractor.children())[-1].weight
        self.feature_extractor = torch.nn.Sequential(*removed)
        features = self.feature_extractor(data)
        out_mean = features @ self.ll_mu.T
        out_cov = features @ self.cov_approx @ torch.transpose(features, 0, 1)
        return out_mean, out_cov

  