from enum import Enum
from turtle import forward
from typing import List

import torch
from torch import nn
from src.log_likelihood import LogLikelihood
from src.network.feature_extractor import FC_Net
from src.utils.TrainWrapper import Trainwrapper
from src.derivatives import nth_derivative, gradient_magnitude
from src.log_likelihood.Regression import RegressionNoNoise
from backpack import backpack, extend
from backpack.extensions import DiagHessian, DiagGGNExact
from torch.autograd import grad
from torch.utils.data import DataLoader


class LaplaceVI(nn.Sequential):
    """Base class for Last-Layer Variational Inference Networks (LLVI).
    """


    def __init__(self, feature_dim: int, out_dim: int, feature_extractor:nn.Sequential, prior_log_var:int=0, tau:int=1e-2, lr: int = 1e-2, optimizer_type: torch.optim.Optimizer = torch.optim.SGD, wdecay=0) -> None:
        ll_model = nn.Sequential(nn.Linear(feature_dim, out_dim, bias=False))
        super().__init__(*feature_extractor.children(), *ll_model.children())
        self.feature_extractor = feature_extractor
        self.ll_model = ll_model
        self.loss_fun = torch.nn.MSELoss(reduction="mean")
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.tau = tau
        self.lr = lr
        self.prior_log_var = torch.tensor([prior_log_var])
        # self.feature_extractor_optimizer = optimizer_type(self.feature_extractor.parameters(), lr=lr, weight_decay=wdecay)
        # self.ll_model_optimizer = optimizer_type(self.ll_model.parameters(), lr=lr)
        self.optimizer = optimizer_type(list(self.feature_extractor.parameters()) + list(self.ll_model.parameters()), lr=lr, weight_decay=wdecay)
        self.covariance_matrix = torch.zeros_like(ll_model[-1].weight)
        extend(self)
        extend(self.loss_fun)

    def ggn(self, features, prediction, target):
        return (torch.square(prediction - target) * features.square()).sum(dim=0)

    def cov(self, features, prediction, target):
        return 1/self.ggn(features, prediction, target) / features.shape[0]

    def full_cov_backpack(self, dataloader:DataLoader, n_datapoints):
        self.optimizer.zero_grad()
        for data, target in dataloader:
            with backpack(DiagGGNExact()):
                self.neg_log_joint(self.ll_model(self.feature_extractor(data)), target, n_datapoints).backward()
        cov = 1/self.ll_model[-1].weight.diag_ggn_exact
        cov/= n_datapoints
        return cov

    def full_cov(self, dataloader:DataLoader, n_datapoints):
        cov = torch.zeros_like(self.ll_model[-1].weight)
        for data, target in dataloader:
            features = self.feature_extractor(data)
            prediction = self.ll_model(features)
            cov += self.ggn(features, prediction, target)
        cov = 1/cov
        cov/=n_datapoints
        return cov


    def neg_log_joint(self, prediction, target, n_datapoints):
        neg_log_lik = self.loss_fun(prediction, target)
        # neg_log_prior = self.get_neg_log_prior(self.ll_mu, n_datapoints)
        # we leave the prior distribution to the wdecay term in the optimizer, this ensures consistency
        return neg_log_lik #+ neg_log_prior

    def get_neg_log_prior(self, weights:torch.Tensor, n_datapoints:int):
        return self.tau * 0.5 * weights.square().sum()/torch.exp(self.prior_log_var)/n_datapoints

    def step_vi(self, data, target, n_datapoints):
        # clear gradients
        self.optimizer.zero_grad()
        # forward pass
        features = self.feature_extractor(data)
        prediction = self.ll_model(features)
        # backward pass
        loss = self.neg_log_joint(prediction, target, n_datapoints)
        loss.backward(retain_graph=True)
        # hessian backward pass, we need to scale by total number of datapoints
        hessian = self.ggn(features, prediction, target)
        neg_entropy = hessian.log().sum()
        self.ll_model[-1].weight.grad += self.tau * grad(neg_entropy, self.ll_model[-1].weight)[0] / n_datapoints
        # step
        self.optimizer.step()
        return loss, -self.tau * neg_entropy /n_datapoints

    def train_without_VI(self, train_loader, epochs, n_datapoints, callback=None):
        self.train()
        train_wrapper = Trainwrapper(["loss"])
        def train_step_fun(data, target):
            # clear gradients
            self.optimizer.zero_grad()
            features = self.feature_extractor(data)
            prediction = self.ll_model(features)
            # compute loss functions
            loss = self.neg_log_joint(prediction, target, n_datapoints)
            # backward pass
            loss.backward()
            self.optimizer.step()
            return [loss]

        output = train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = None,  callback=callback)
        # now set cov
        self.covariance_matrix = self.full_cov(train_loader, n_datapoints)
        self.covariance_matrix_backpack = self.full_cov_backpack(train_loader, n_datapoints).squeeze().diag()
        return output


    def train_model(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, callback=None):
        self.train()
        train_wrapper = Trainwrapper(["loss", "scaled_entropy"])
        step_fun = lambda x,y: self.step_vi(x, y, n_datapoints)
        output = train_wrapper.wrap_train(epochs, train_loader, step_fun, train_hyper_fun = None, hyper_update_step=update_freq, callback=callback)
        # now set cov
        self.covariance_matrix = self.full_cov(train_loader, n_datapoints)
        self.covariance_matrix_backpack = self.full_cov_backpack(train_loader, n_datapoints).squeeze().diag()
        return output

    def train_model_with_trajectories(self, train_loader, n_datapoints, epochs=1):
        self.train()
        train_wrapper = Trainwrapper(["loss", "entropy"])
        step_fun = lambda x,y: self.step_vi(x, y, n_datapoints)
        self.trajectories_mean = torch.empty_like(self.ll_model[-1].weight)
        self.trajectories_cov = torch.empty_like(self.ll_model[-1].weight)
        def save_trajectories(*args):
            with torch.no_grad():
                self.trajectories_mean = torch.cat((self.trajectories_mean, self.ll_model[-1].weight.detach()), dim=0)
                self.trajectories_cov = torch.cat((self.trajectories_cov, self.full_cov(train_loader, n_datapoints)), dim=0)
            
        
        output = train_wrapper.wrap_train(epochs, train_loader, step_fun, callback=save_trajectories)
        # now set cov
        self.covariance_matrix = self.full_cov(train_loader, n_datapoints)
        self.covariance_matrix_backpack = self.full_cov_backpack(train_loader, n_datapoints).squeeze().diag()
        return output, self.trajectories_mean[1:, :], self.trajectories_cov[1:, :]


    def train_full_trajectories(self, train_loader, n_datapoints, epochs=1):
        raise NotImplementedError
        self.train()
        train_wrapper = Trainwrapper(["loss", "entropy"])
        step_fun = lambda x,y: self.step_vi(x, y, n_datapoints)
        self.trajectories_mean = torch.empty_like(self.ll_model[-1].weight)
        self.trajectories_cov = torch.empty_like(self.ll_model[-1].weight)
        def save_trajectories(*args):
            with torch.no_grad():
                self.trajectories_mean = torch.cat((self.trajectories_mean, self.ll_model[-1].weight.detach()), dim=0)
                self.trajectories_cov = torch.cat((self.trajectories_cov, self.full_cov(train_loader, n_datapoints)), dim=0)
            
        
        output = train_wrapper.wrap_train(epochs, train_loader, step_fun, callback=save_trajectories)
        # now set cov
        self.covariance_matrix = self.full_cov(train_loader, n_datapoints)
        self.covariance_matrix_backpack = self.full_cov_backpack(train_loader, n_datapoints).squeeze().diag()
        return output, self.trajectories_mean[1:, :], self.trajectories_cov[1:, :]


    def predict(self, data):
        features = self.feature_extractor(data)
        out_mean = self.ll_model(features)
        out_cov = features @ self.covariance_matrix.squeeze().diag() @ torch.transpose(features, 0, 1)
        return out_mean, out_cov

    def forward(self, data):
        return self.ll_model(self.feature_extractor(data))
