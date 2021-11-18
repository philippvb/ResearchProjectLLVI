import os
from datetime import datetime
from enum import Enum
from typing import List

import torch
from torch import nn
from src.log_likelihood import LogLikelihood
from src.network.feature_extractor import FC_Net
from src.utils.TrainWrapper import Trainwrapper
from src.weight_distribution import WeightDistribution


class LikApprox(Enum):
    """Available Likelihood approximations
    """
    MAXIMUMLIKELIHOOD = "ML"
    MONTECARLO = "MC"
    CLOSEDFORM = "CF"
    PROBIT = "PR"
    

class LLVINetwork(nn.Module):
    """Base class for Last-Layer Variational Inference Networks (LLVI).
    """


    def __init__(self, feature_dim: int, out_dim: int, feature_extractor:FC_Net, weight_dist: WeightDistribution, loss_fun:LogLikelihood, prior_mu:int=0, prior_log_var:int=0, tau:int=1e-2, lr: int = 1e-2, optimizer_type: torch.optim.Optimizer = torch.optim.Adam) -> None:
        super().__init__()
        self.loss_fun = loss_fun
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.feature_extractor = feature_extractor
        self.tau = tau
        self.lr = lr


        self.prior_mu: torch.Tensor = self.create_full_tensor(feature_dim, out_dim, prior_mu)
        self.prior_log_var: torch.Tensor = self.create_full_tensor(feature_dim, out_dim, prior_log_var)
        self.prior_optimizer: torch.optim.Optimizer = self.init_hyperparam_optimizer(optimizer_type)

        self.weight_distribution = weight_dist

        self.nn_optimizers: List[torch.optim.Optimizer] = [feature_extractor.optimizer, self.weight_distribution.optimizer] # optimizers of the neural network


    # utils functions

    def init_hyperparam_optimizer(self, optimizer_type: torch.optim.Optimizer):
        return optimizer_type([self.prior_mu, self.prior_log_var], self.lr)


    def create_full_tensor(self, dim1, dim2, fill_value):
        return nn.Parameter(torch.full((dim1, dim2), fill_value=fill_value, requires_grad=True, dtype=torch.float32))

    def save(self, filedir):
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filedir += "/" + timestamp
        os.makedirs(filedir)
        # save model
        model_filename = "/model.pt"
        torch.save(self.state_dict(), filedir + model_filename)
        # save config
        self.save_config(filedir)
        return filedir

    def load(self, filedir):
        self.load_state_dict(torch.load(filedir + "/model.pt"))

    def update_prior_mu(self, value):
        self.prior_mu = nn.Parameter(value.detach().clone())

    def clear_nn_optimizers(self):
        [optimizer.zero_grad() for optimizer in self.nn_optimizers]

    def step_nn_optimizers(self):
        [optimizer.step() for optimizer in self.nn_optimizers]

    # forward pass and prediction

    def forward_MC(self, x:torch.Tensor, samples=10) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = features @ self.sample_ll(samples=samples)
        return output

    def forward_ML(self, x: torch.Tensor) -> torch.Tensor:
        """Computes a forward pass with the ML estimate of the Last-Layer weights,
            (in case of Gaussian the Mean)

        Args:
            x (torch.Tensor): The input data in shape batch_size x feature_dims...

        Returns:
            (torch.Tensor): The log likelihood of class predictions
        """
        features = self.feature_extractor(x)
        output = features @ self.get_ll_mu()
        return output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        pred_cov = features @ self.get_ll_cov() @ torch.transpose(features, 0, 1)
        pred_mean = features @ self.get_ll_mu()
        return pred_mean, pred_cov

    # last-layer approximation
    def get_ll_mu(self) -> torch.Tensor:
        return self.weight_distribution.get_mu()

    def get_ll_cov(self) -> torch.Tensor:
        return self.weight_distribution.get_cov()

    def sample_ll(self, samples=10):
        return self.weight_distribution.sample(samples=samples)

    def KL_div(self) -> torch.Tensor:
        return self.weight_distribution.KL_div(self.prior_mu, self.prior_log_var)


    # training:
    def compute_prediction_loss(self, data: torch.Tensor, target: torch.Tensor, method: LikApprox, **method_kwargs) -> torch.Tensor:
        if method == LikApprox.MAXIMUMLIKELIHOOD:
            prediction = self.forward_ML(data)
            return self.loss_fun(prediction, target, average=False)
        elif method == LikApprox.MONTECARLO:
            prediction = self.forward_MC(data)
            return self.loss_fun(prediction, target, average=True)
        else:
            raise ValueError(f"Method {method} not implemented")


    def train_without_VI(self, train_loader, epochs=1):
        # self.model_config.update({"trained without VI, epochs": epochs})
        self.train()
        train_wrapper = Trainwrapper(["prediction_loss"])

        def train_step_fun(data, target):
            # clear gradients
            self.clear_nn_optimizers()
            # compute loss functions
            loss = self.compute_prediction_loss(data, target, method=LikApprox.MAXIMUMLIKELIHOOD)
            # backward pass
            loss.backward()
            self.step_nn_optimizers()
            return [loss]

        return train_wrapper.wrap_train(epochs=epochs, train_loader=train_loader, train_step_fun=train_step_fun)

    # train functions
    def train_model(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
        # self.model_config.update({"trained model only, epochs": epochs, "train_samples": samples, "train_hyper": train_hyper, "hyper update freq": update_freq})
        self.train()
        train_wrapper = Trainwrapper(["prediction_loss", "kl_loss"])

        def train_step_fun(data, target):
            # clear gradients
            self.clear_nn_optimizers()
            # compute loss functions
            prediction_loss = self.compute_prediction_loss(data, target, method, **method_kwargs)
            kl_loss = self.tau * self.KL_div() / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            # backward pass
            loss.backward()
            self.step_nn_optimizers()
            return [prediction_loss, kl_loss]

        hyper_fun = lambda train_loader: self.train_hyper_epoch(train_loader, n_datapoints, method, **method_kwargs) if train_hyper else None

        return train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=update_freq)


    def train_hyper_epoch(self, train_loader:torch.Tensor, n_datapoints:int, method:LikApprox, **method_kwargs) -> None:
        """Trains the hyperparameters/prior parameters of the VI model for one epoch

        Args:
            train_loader ([type]): Trainingdata loader
            n_datapoints ([type]): number of datapoints in train_loader for KL Scaling
            samples ([type]): How many samples to take for each forward pass
        """
        for batch_idx, (data, target) in enumerate(train_loader):
            self.prior_optimizer.zero_grad()
            prediction_loss = self.compute_prediction_loss(data, target, method, **method_kwargs)
            kl_loss = self.tau * self.KL_div() / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            loss.backward()
            self.prior_optimizer.step()


    
