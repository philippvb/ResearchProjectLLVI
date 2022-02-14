from enum import Enum
from turtle import forward
from types import MethodDescriptorType
from typing import List

import torch
from torch import nn
from src.log_likelihood import LogLikelihood
from src.network.feature_extractor import FC_Net
from src.utils.TrainWrapper import Trainwrapper
from src.derivatives import nth_derivative, gradient_magnitude
from src.log_likelihood.Regression import RegressionNoNoise

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

        self.ll_mu = nn.Parameter(torch.randn(feature_dim, out_dim, requires_grad=True))
        self.ll_optimizer = optimizer_type([self.ll_mu], lr)

        self.nn_optimizers: List[torch.optim.Optimizer] = [feature_extractor.optimizer, self.ll_optimizer] # optimizers of the neural network

    # def create_ll_optimizer(self, optim: torch.optim.Optimizer, **optim_kwargs):
    #     return optim([self.prior_mu], **optim_kwargs)

    def create_full_tensor(self, dim1, dim2, fill_value):
        return nn.Parameter(torch.full((dim1, dim2), fill_value=fill_value, requires_grad=True, dtype=torch.float32))

    def compute_loss_wrt_ll(self, features, target, ll_weights):
        prediction = features @ ll_weights
        neg_log_lik = self.loss_fun(prediction, target)
        log_prior = self.get_log_prior(ll_weights)
        return neg_log_lik - log_prior

    def compute_diag_hess(self, data, target):
        with torch.no_grad():
            features = self.feature_extractor(data)
        f= lambda x: self.compute_loss_wrt_ll(features, target, x)
        hess = torch.autograd.functional.hessian(f, self.ll_mu, create_graph=True)
        hess = torch.diagonal(hess)
        hess = hess.reshape(self.ll_mu.shape)
        return hess

    def compute_entropy_grad(self, data, target):
        hess = self.compute_diag_hess(data, target)
        entropy = torch.log(hess).sum()
        entropy_grad = torch.autograd.grad(entropy, self.ll_mu, create_graph=True)[0]
        return entropy_grad

    def compute_neg_log_joint(self, data, target):
        features = self.feature_extractor(data)
        prediction = features @ self.ll_mu
        neg_log_lik = self.loss_fun(prediction, target)
        log_prior = self.get_log_prior(self.ll_mu)
        return neg_log_lik - log_prior

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


    def train_model(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, callback=None, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
        self.train()
        train_wrapper = Trainwrapper(["loss"])
        def train_step_fun(data, target):
            # clear gradients
            self.clear_nn_optimizers()
            # compute loss functions
            loss = self.compute_neg_log_joint(data, target)
            # backward pass
            loss.backward()
            self.ll_mu.grad += 0.1 * self.compute_entropy_grad(data, target)
            self.step_nn_optimizers()
            return [loss]

        # hyper_fun = lambda train_loader: self.train_hyper_epoch(train_loader, n_datapoints, method, **method_kwargs) if train_hyper else None
        hyper_fun=None

        output = train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=update_freq, callback=callback)
        # now set cov
        self.set_cov(train_loader)
        return output

    def set_cov(self, dataloader):
        self.cov_approx = torch.zeros_like(self.ll_mu)
        for data, target in dataloader:
            self.cov_approx += 1/self.compute_diag_hess(data, target)
            # self.cov_approx += self.get_cov_gm(self.compute_neg_log_joint(data, target))
        self.cov_approx/=512
        self.cov_approx = torch.diag(self.cov_approx.flatten())
        # print("Cov approx is")
        # print(self.cov_approx)


    def forward(self, data):
        features = self.feature_extractor(data)
        out_mean = features @ self.ll_mu
        out_cov = features @ self.cov_approx @ torch.transpose(features, 0, 1)
        return out_mean, out_cov

  