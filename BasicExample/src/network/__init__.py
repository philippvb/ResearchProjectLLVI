from abc import abstractmethod
import os
from datetime import datetime
from enum import Enum
from typing import List
from numpy import dtype

import torch
from torch import nn
from src.log_likelihood import LogLikelihood
from src.network.feature_extractor import FC_Net
from src.utils.TrainWrapper import Trainwrapper
from src.weight_distribution import WeightDistribution
import pandas as pd
from tqdm import tqdm

class LikApprox(Enum):
    """Available Likelihood approximations
    """
    MAXIMUMLIKELIHOOD = "ML"
    MONTECARLO = "MC"
    CLOSEDFORM = "CF"

class PredictApprox(Enum):
    MAXIMUMLIKELIHOOD = "ML"
    MONTECARLO = "MC"
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

    # def update_prior_mu(self, value):
    #     self.prior_mu = nn.Parameter(value.detach().clone())

    def clear_nn_optimizers(self):
        [optimizer.zero_grad() for optimizer in self.nn_optimizers]

    def step_nn_optimizers(self):
        [optimizer.step() for optimizer in self.nn_optimizers]

    # forward pass and prediction

    def forward_MC(self, x:torch.Tensor, samples=10) -> torch.Tensor:
        features = self.feature_extractor(x)
        output = features @ self.sample_ll(samples=samples)
        return output

    def forward_MC2(self, x:torch.Tensor, samples=10):
        mean, cov = self.forward_multi(x)
        pred_samples = torch.distributions.MultivariateNormal(mean, cov).sample((samples, ))
        return pred_samples


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

    @abstractmethod
    def forward(self, x: torch.Tensor, method: PredictApprox):
        raise NotImplementedError

    def forward_single(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        pred_mean = features @ self.get_ll_mu()
        pred_cov = features @ self.get_ll_cov() @ torch.transpose(features, 0, 1)
        return pred_mean, pred_cov

    def forward_multi(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_datapoints = x.shape[0]
        features = self.feature_extractor(x)
        pred_mean = features @ self.get_ll_mu()
        features_concat = torch.zeros((n_datapoints, self.out_dim, self.feature_dim * self.out_dim))
        for i in range(self.out_dim):
            features_concat[:, i, i*self.feature_dim:(i+1) * self.feature_dim] = features
        ll_cov = self.get_ll_cov()
        # print(ll_cov)
        pred_cov = features_concat @ ll_cov @ torch.transpose(features_concat, dim0=1, dim1=-1)
        wrong = torch.Tensor([torch.equal(pred_cov[i], torch.zeros((2,2))) for i in range(n_datapoints)]).bool()
        # print(x[wrong])
        # print(features[wrong])
        # print(self.feature_extractor(x[wrong]))
        return pred_mean, pred_cov

    def forward_log_marginal_likelihood(self, x: torch.Tensor, samples:int) -> tuple[torch.Tensor, torch.Tensor]:
        n_datapoints = x.shape[0]
        features = self.feature_extractor(x)
        pred_mean = features @ self.prior_mu
        features_concat = torch.zeros((n_datapoints, self.out_dim, self.feature_dim * self.out_dim))
        for i in range(self.out_dim):
            features_concat[:, i, i*self.feature_dim:(i+1) * self.feature_dim] = features
        ll_cov = torch.diag(torch.flatten(torch.exp(self.prior_log_var)))
        pred_cov = features_concat @ ll_cov @ torch.transpose(features_concat, dim0=1, dim1=-1)
        pred_samples = torch.cat([(torch.cholesky(pred_cov[i]) @ torch.randn((self.out_dim, samples))).T for i in range(n_datapoints)]).reshape((samples, n_datapoints, self.out_dim))
        return pred_samples



    def forward_multi_no_pad(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise ValueError("Not implemented yet")
        features = self.feature_extractor(x)
        pred_mean = features @ self.get_ll_mu()
        ll_cov = torch.reshape(self.get_ll_cov(), (self.out_dim, self.out_dim, self.feature_dim, self.feature_dim))
        pred_cov1 = (features @ ll_cov)#.permute(1, 3, 2, 0)
        pred_cov = (pred_cov1 @ features.T).permute(3,2,1,0)
        pred_cov = torch.stack([pred_cov[i,i] for i in range(x.shape[0])], dim=0)
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
    # this function is stricly for training, not for predicting!!!
    def compute_prediction_loss(self, data: torch.Tensor, target: torch.Tensor, method: LikApprox, **method_kwargs) -> torch.Tensor:
        if method == LikApprox.MAXIMUMLIKELIHOOD:
            prediction = self.forward_ML(data)
            return self.loss_fun(prediction, target, average=False)
        elif method == LikApprox.MONTECARLO:
            prediction = self.forward_MC(data, **method_kwargs)
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
    def train_model(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, callback=None, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
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

        return train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=update_freq, callback=callback)

        # train functions
    def train_ll_only(self, train_loader, n_datapoints, epochs=1, train_hyper=False, update_freq=10, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
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
            self.weight_distribution.optimizer.step()
            return [prediction_loss, kl_loss]

        hyper_fun = lambda train_loader: self.train_hyper_epoch(train_loader, n_datapoints, method, **method_kwargs) if train_hyper else None

        return train_wrapper.wrap_train(epochs, train_loader, train_step_fun, train_hyper_fun = hyper_fun, hyper_update_step=update_freq)

    def train_feature_extractor_only(self, train_loader, epochs=1):
        self.train()
        train_wrapper = Trainwrapper(["prediction_loss"])

        def train_step_fun(data, target):
            # clear gradients
            self.feature_extractor.optimizer.zero_grad()
            # compute loss functions
            loss = self.compute_prediction_loss(data, target, method=LikApprox.MAXIMUMLIKELIHOOD)
            # backward pass
            loss.backward()
            self.feature_extractor.optimizer.step()
            return [loss]

        return train_wrapper.wrap_train(epochs, train_loader, train_step_fun)


    def train_em_style(self, train_loader, n_datapoints, total_epochs=10, inner_epochs_fe=10, inner_epochs_vi=10, method:LikApprox=LikApprox.MONTECARLO, **method_kwargs):
        self.train()

        loss_tracker = pd.DataFrame(columns={"Feature extractor loss", "Log_Lik", "KL_loss"})
        pbar = tqdm(range(total_epochs))
        for epoch in pbar:
            # E step: do the feature extractor
            for i in range(inner_epochs_fe):
                batch_loss = torch.zeros(inner_epochs_fe)
                for batch_id, (data, target) in enumerate(train_loader):
                    # clear gradients
                    self.feature_extractor.optimizer.zero_grad()
                    # compute loss functions
                    loss = self.compute_prediction_loss(data, target, method=LikApprox.MAXIMUMLIKELIHOOD)
                    # backward pass
                    loss.backward()
                    self.feature_extractor.optimizer.step()
                    batch_loss[i] = loss

            # M step: do the last layer
            for i in range(inner_epochs_vi):
                batch_log_lik = torch.zeros(inner_epochs_vi)
                batch_kl_loss = torch.zeros(inner_epochs_vi)
                for batch_id, (data, target) in enumerate(train_loader):
                    # clear gradients
                    self.weight_distribution.optimizer.zero_grad()
                    # compute loss functions
                    prediction_loss = self.compute_prediction_loss(data, target, method, **method_kwargs)
                    kl_loss = self.tau * self.KL_div() / n_datapoints # rescale kl_loss
                    loss = prediction_loss + kl_loss
                    # backward pass
                    loss.backward()
                    self.weight_distribution.optimizer.step()
                    batch_log_lik[i] = prediction_loss
                    batch_kl_loss[i] = kl_loss


            batch_loss = batch_loss.detach().clone()
            batch_kl_loss = batch_kl_loss.detach().clone()
            batch_log_lik = batch_log_lik.detach().clone()
            description = f"FE:{round(batch_loss.mean().item(), 2)}, LogLik:{round(batch_log_lik.mean().item(), 2)}, KL_div:{round(batch_kl_loss.mean().item(), 2)}"
            pbar.set_description(description)
            loss_tracker = loss_tracker.append(pd.DataFrame.from_dict({"Feature extractor loss": [batch_loss.mean().item()], "Log_Lik": [batch_log_lik.mean().item()], "KL_loss": [batch_kl_loss.mean().item()]})) 

        return loss_tracker

        





    def train_hyper(self, train_loader, epochs:int, samples:int):
        self.train()
        train_wrapper = Trainwrapper(["prediction_loss"])
        print(self.prior_mu, self.prior_log_var)

        def train_step_fun(data, target):
            # clear gradients
            self.prior_optimizer.zero_grad()
            # compute loss functions
            prediction = self.forward_log_marginal_likelihood(data, samples=samples)
            prediction_loss = self.loss_fun(prediction, target, average=True)
            # backward pass
            prediction_loss.backward()
            self.prior_optimizer.step()
            return [prediction_loss]
        
        return train_wrapper.wrap_train(epochs, train_loader, train_step_fun)


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


    
