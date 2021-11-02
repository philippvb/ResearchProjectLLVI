import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from enum import Enum
from datetime import datetime
import os
import math

class Loss(str, Enum):
    CATEGORICAL = "categorical"
    MSE = "mse"



class LLVI_network(nn.Module):
    """Base class for Last-Layer Variational Inference Networks (LLVI).
    """
    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, lr=1e-2, tau=1, wdecay=0, bias=True, loss=Loss.CATEGORICAL, data_log_var=-1) -> None:
        super(LLVI_network, self).__init__()
        self.bias = bias
        if self.bias:
            feature_dim += 1
        self.feature_extractor: nn.Module = feature_extractor
        if loss == Loss.CATEGORICAL:
            self.loss_fun = self.loss_fun_categorical
        elif loss == Loss.MSE:
            self.loss_fun = self.loss_fun_regression

        self.tau = tau
        self.feature_extractor_optimizer = optim.SGD(self.feature_extractor.parameters(), lr=lr, momentum=0.8, weight_decay=wdecay)

        self.prior_mu = nn.Parameter(torch.full((feature_dim, out_dim), fill_value=prior_mu, requires_grad=True, dtype=torch.float32))
        self.prior_log_var = nn.Parameter(torch.full((feature_dim, out_dim), fill_value=prior_log_var, requires_grad=True, dtype=torch.float32))
        self.data_log_var = nn.Parameter(torch.tensor([data_log_var], dtype=torch.float32), requires_grad=True) # the log variance of the data
        self.prior_optimizer = optim.SGD([self.prior_mu, self.prior_log_var, self.data_log_var], lr=lr, momentum=0.8) # optimizer for prior

    def save(self, filedir):
        if not os.path.exists(filedir, exist_ok=True):
            os.makedirs(filedir)
        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        filename = f"/{self.__class__.__name__}_{timestamp}.pt"
        torch.save(self.state_dict(), filedir + filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    def loss_fun_categorical(self, pred, target, mean=True):
        output = F.log_softmax(pred, dim=-1) # convert to logprobs
        if mean:
            output = torch.mean(output, dim=0) # take the mean
        return F.nll_loss(output, target, reduction="mean")

    def loss_fun_regression(self, pred, target, mean=True):
        """Returns the negative log likelihood of the target (true) data given
        the prediction in the gaussian case/regression. Scaled by 1/datapoints.

        Args:
            pred (torch.Tensor): the prediction
            target (torch.Tensor): the target data
            mean (bool, optional): If true, assumes multiple predictions in dim 0 and averages over them. Defaults to True.

        Returns:
            torch.Tensor: The negative log-likelihood
        """
        if mean:
            pred = torch.mean(pred, dim=0) # take the mean
        squared_diff = F.mse_loss(pred, target)
        return 0.5 * torch.log(2 * math.pi * torch.exp(0.5 * self.data_log_var)) + 0.5 * squared_diff / torch.exp(self.data_log_var)

    def forward(self, x, samples=1):
        features = self.feature_extractor(x)
        if self.bias: # add bias of ones
            features = self.add_bias(features)
        output = features @ self.sample_ll(samples=samples)
        kl_loss = self.KL_div()
        return output, kl_loss

    def add_bias(self, features):
        """Adds a bias feature to the end of a tensor

        Args:
            features (torch.tensor): The features of size batch_size x feature_dims

        Returns:
            [torch.tensor]: features with added tensor of ones at the end, size batch_size x feature_dims+1
        """
        bias_tensor = torch.ones((features.shape[0], 1))
        return torch.cat((features, bias_tensor),dim=-1)



    def forward_train_LL(self, x, samples=1):
        """Forward pass for Training just the last layer

        Args:
            x (torch.Tensor): The input data in shape batch_size x feature_dims...
            samples (int, optional): Number of samples to take from the last-layer weight distribution. Defaults to 1.

        Returns:
            tuple(torch.Tensor, torch.Tensor): The log likelihood of class predictions and the Kl divergence
        """
        with torch.no_grad():
            features = self.feature_extractor(x)
            if self.bias:
                features = self.add_bias(features)
        output = features @ self.sample_ll(samples=samples)
        kl_loss = self.KL_div()
        return output, kl_loss

    def forward_ML_estimate(self, x):
        """Computes a forward pass with the ML estimate of the Last-Layer weights,
            (in case of Gaussian the Mean)

        Args:
            x (torch.Tensor): The input data in shape batch_size x feature_dims...

        Returns:
            (torch.Tensor): The log likelihood of class predictions
        """
        features = self.feature_extractor(x)
        if self.bias:
            features = self.add_bias(features)
        output = features @ self.ll_mu
        return output

    def sample_ll(self, samples=1):
        raise NotImplementedError

    def KL_div(self):
        raise NotImplementedError

    def train_wrapper(self, epochs, train_loader, train_hyper, update_freq, n_datapoints, samples, train_step_fun):
        self.train()
        epoch_losses = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            batch_kl_losses = torch.zeros(len(train_loader))
            batch_prediction_losses = torch.zeros_like(batch_kl_losses)
            for batch_id, (data, target) in enumerate(train_loader):
                # forward pass
                prediction_loss, kl_loss = train_step_fun(data, target)
                # logging
                with torch.no_grad():
                    batch_kl_losses[batch_id] = kl_loss.item()
                    batch_prediction_losses[batch_id] = prediction_loss.item()

            # update the hyperparameters
            if train_hyper and (epoch % update_freq) == 0:
                self.train_hyper(train_loader, n_datapoints, samples)

            current_epoch_loss = torch.mean(kl_loss + prediction_loss)
            digits = 2
            pbar.set_description(f"Loss:{round(current_epoch_loss.item(), digits)}, PredLoss:{round(torch.mean(prediction_loss).item(), digits)}, KLLoss:{round(torch.mean(kl_loss).item(), digits)}")
            epoch_losses.append(current_epoch_loss)
        return epoch_losses


    def train_model(self, train_loader, n_datapoints, epochs=1, samples=1, train_hyper=False, update_freq=10):
        def train_step_fun(data, target):
            # clear gradients
            self.ll_optimizer.zero_grad()
            self.feature_extractor_optimizer.zero_grad()
            # compute loss functions
            output, kl_loss = self.forward(data, samples=samples)
            prediction_loss = self.loss_fun(output, target)
            kl_loss = self.tau * kl_loss / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            # backward pass
            loss.backward()
            self.ll_optimizer.step()
            self.feature_extractor_optimizer.step()
            return prediction_loss, kl_loss

        return self.train_wrapper(epochs=epochs, train_loader=train_loader, n_datapoints=n_datapoints, samples=samples, train_step_fun=train_step_fun, train_hyper=train_hyper, update_freq=update_freq)


    def train_hyper(self, train_loader, n_datapoints, samples):
        """Trains the hyperparameters/prior parameters of the VI model

        Args:
            train_loader ([type]): Trainingdata loader
            n_datapoints ([type]): number of datapoints in train_loader for KL Scaling
            samples ([type]): How many samples to take for each forward pass
        """
        for batch_idx, (data, target) in enumerate(train_loader):
            self.prior_optimizer.zero_grad()
            output, kl_loss = self.forward(data, samples=samples)
            prediction_loss = self.loss_fun(output, target)
            kl_loss = self.tau * kl_loss / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            loss.backward()
            self.prior_optimizer.step()


    def train_without_VI(self, train_loader, epochs=1):
        def train_step_fun(data, target):
            # clear gradients
            self.ll_optimizer.zero_grad()
            self.feature_extractor_optimizer.zero_grad()
            # compute loss functions
            output = self.forward_ML_estimate(data)
            prediction_loss = self.loss_fun(output, target, mean=False)
            loss = prediction_loss
            # backward pass
            loss.backward()
            self.ll_optimizer.step()
            self.feature_extractor_optimizer.step()
            return prediction_loss, torch.tensor([0], dtype=torch.float32)

        return self.train_wrapper(epochs=epochs, train_loader=train_loader, n_datapoints=-1, samples=-1, train_step_fun=train_step_fun, train_hyper=False, update_freq=-1)


    def train_LL(self, train_loader, n_datapoints, epochs=1, samples=1, train_hyper=False, update_freq=10):
        def train_step_fun(data, target):
            # clear gradients
            self.ll_optimizer.zero_grad()
            # compute loss functions
            output, kl_loss = self.forward_train_LL(data, samples=samples)
            prediction_loss = self.loss_fun(output, target)
            kl_loss = self.tau * kl_loss / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            # backward pass
            loss.backward()
            self.ll_optimizer.step()
            return prediction_loss, kl_loss

        return self.train_wrapper(epochs=epochs, train_loader=train_loader, n_datapoints=n_datapoints, samples=samples, train_step_fun=train_step_fun, train_hyper=train_hyper, update_freq=update_freq)



    def test(self, test_loader, samples=5):
        test_losses = []
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output, kl_div = self.forward(data, samples=samples)
                test_loss += self.loss_fun(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        return test_losses

    def test_confidence(self, test_loader, ood_test_loader=None, samples=5):
        self.eval()
        confidence_batch = []
        with torch.no_grad():
            for data, target in test_loader:
                output, kl_div = self.forward(data, samples=samples)
                output_probs = torch.exp(output)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for in distribution data is: {sum(confidence_batch)/len(confidence_batch)}")

        ood_confidence_batch = []
        with torch.no_grad():
            for data, target in ood_test_loader:
                output, kl_div = self.forward(data, samples=5)
                output_probs = torch.exp(output)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                ood_confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for out-of distribution data is: {sum(ood_confidence_batch)/len(ood_confidence_batch)}")


class LLVI_network_diagonal(LLVI_network):

    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_log_var=0, lr=1e-2, tau=1, wdecay=0, bias=True, loss=Loss.CATEGORICAL) -> None:
        super(LLVI_network_diagonal, self).__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau, wdecay=wdecay, bias=bias, loss=loss)
        
        if self.bias:
            feature_dim += 1
        self.ll_mu = nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim), requires_grad=True)
        self.ll_log_var = nn.Parameter(init_ll_log_var + torch.randn_like(self.ll_mu), requires_grad=True)
        self.ll_optimizer = optim.SGD([self.ll_mu, self.ll_log_var],lr=lr,momentum=0.8)
        self.ll_n_parameters = feature_dim * out_dim

    def sample_ll(self, samples=1):
        std = torch.multiply(torch.exp(0.5 * self.ll_log_var),  torch.randn((samples, ) + self.ll_log_var.size()))
        return self.ll_mu + std

    def KL_div(self):
        div = 0.5 * (torch.sum(self.prior_log_var) - torch.sum(self.ll_log_var) - self.ll_n_parameters + torch.sum(torch.exp(self.ll_log_var - self.prior_log_var)) + torch.sum(torch.div(torch.square(self.prior_mu - self.ll_mu), torch.exp(self.prior_log_var))))
        return div

    def KL_div_torch(self):
        q_std = torch.diag(torch.flatten(torch.exp(self.ll_log_var)))
        q_mu = torch.flatten(self.ll_mu)
        q = torch.distributions.MultivariateNormal(q_mu, q_std)
        p_std = torch.diag(torch.flatten(torch.exp(self.prior_log_var)))
        p_mu = torch.flatten(self.prior_mu)
        p = torch.distributions.MultivariateNormal(p_mu, p_std)
        return torch.distributions.kl_divergence(q, p)


class LLVI_network_KFac(LLVI_network):
    def __init__(self, feature_extractor, feature_dim, out_dim, A_dim, B_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_cov_scaling=1, lr=1e-2, tau=1, wdecay=0, bias=True, loss=Loss.CATEGORICAL) -> None:
        super(LLVI_network_KFac, self).__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau, wdecay=wdecay, bias=bias, loss=loss)

        # feature dimensions
        if self.bias:
            feature_dim += 1
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.ll_n_parameters = feature_dim * out_dim

        # last-layer mean 
        self.ll_mu =  nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim, requires_grad=True))
        # cholesky decomposition of factors
        self.chol_a_lower = nn.Parameter(init_ll_cov_scaling * torch.randn((A_dim, A_dim), requires_grad=True)) # lower triangular matrix without diagonal
        self.chol_a_log_diag =  nn.Parameter(init_ll_cov_scaling * torch.randn(A_dim, requires_grad=True)) # separate diagonal (log since it has to be positive)
        self.chol_b_lower = nn.Parameter(init_ll_cov_scaling * torch.randn((B_dim, B_dim), requires_grad=True))
        self.chol_b_log_diag = nn.Parameter(init_ll_cov_scaling * torch.randn(B_dim, requires_grad=True))
        self.ll_optimizer = optim.SGD([self.ll_mu, self.chol_a_lower, self.chol_a_log_diag, self.chol_b_lower, self.chol_b_log_diag],lr=lr,momentum=0.5) # init optimizer here in oder to get all the parameters in


    def get_ll_cov(self):
        """Create the covariance matrix

        Returns:
            torch.tensor: The covariance matrix
        """
        chol_fac =  self.get_cov_chol()
        return chol_fac @ chol_fac.T

    def get_cov_chol(self):
        """Get the Cholesky decomposition of the Covariance Matrix

        Returns:
            torch.tensor: Lower triangular cholesky decomposition matrix
        """
        a_lower_diag = torch.tril(self.chol_a_lower, diagonal=-1)
        b_lower_diag = torch.tril(self.chol_b_lower, diagonal=-1)
        a = a_lower_diag + torch.diag(torch.exp(self.chol_a_log_diag))
        b = b_lower_diag + torch.diag(torch.exp(self.chol_b_log_diag))
        if not torch.all(torch.diagonal(a)>= 0): # sanity check
            print(torch.exp(self.chol_a_log_diag))
            raise ValueError("Cholesky decomposition of A contains negative values on the diagonal")
        if not torch.all(torch.diagonal(b)>= 0): # sanity check
            print(torch.exp(self.chol_b_log_diag))
            raise ValueError("Cholesky decomposition of B contains negative values on the diagonal")
        return torch.kron(a, b)

    def sample_ll(self, samples=1):
        cov_chol = self.get_cov_chol()
        std = torch.reshape((cov_chol @ torch.randn((self.feature_dim * self.out_dim, samples))).T, (samples, self.feature_dim, self.out_dim))
        return self.ll_mu + std


    def get_log_det(self):
        """Return the log determinant of log(det(sigma_p)/det(sigma_q))

        Returns:
            tensor.float: The value of the logdet
        """
        log_det_q = 2 * torch.sum(torch.log(torch.diagonal(self.get_cov_chol())))
        log_det_p = torch.sum(self.prior_log_var)
        return log_det_p - log_det_q

    def KL_div(self):
        log_determinant = self.get_log_det()
        trace = torch.sum(torch.divide(torch.diagonal(self.get_ll_cov()), torch.flatten(torch.exp(self.prior_log_var))))
        scalar_prod = torch.sum(torch.div(torch.square(self.prior_mu - self.ll_mu), torch.exp(self.prior_log_var)))
        kl_div = 0.5 * (log_determinant - self.ll_n_parameters + trace + scalar_prod)
        if kl_div <= 0: # sanity check
            print("kl_div", kl_div.item())
            print("logdet", log_determinant.item(),"trace", trace.item(),"scalar", scalar_prod.item())
            raise ValueError("KL div is smaller than 0")
        return kl_div




class LLVI_network_full_Cov(LLVI_network):
    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_cov_scaling=1, init_ll_log_var=0, lr=1e-2, tau=1, wdecay=0, bias=False, loss=Loss.CATEGORICAL, data_log_var=-1) -> None:
        super().__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau, wdecay=wdecay, bias=bias, loss=loss, data_log_var=data_log_var)

        if self.bias:
            feature_dim += 1
        self.feature_dim = feature_dim
        self.out_dim = out_dim
        self.ll_n_parameters = feature_dim * out_dim
        # last-layer mean 
        self.ll_mu =  nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim, requires_grad=True))
        # ll covariance
        cov_dim = feature_dim*out_dim
        self.cov_lower = nn.Parameter(init_ll_cov_scaling * (torch.randn((cov_dim, cov_dim), requires_grad=True))) # lower triangular matrix without diagonal
        self.cov_log_diag =  nn.Parameter(init_ll_log_var + torch.randn(cov_dim, requires_grad=True)) # separate diagonal (log since it has to be positive)
        self.ll_optimizer = optim.SGD([self.ll_mu, self.cov_lower, self.cov_log_diag], lr=lr,momentum=0.8) # init optimizer here in oder to get all the parameters in

    def get_cov_chol(self):
        cov_lower_diag = torch.tril(self.cov_lower, diagonal=-1)
        cov = cov_lower_diag + torch.diag(torch.exp(self.cov_log_diag))
        return cov

    def get_ll_cov(self):
        cov_chol = self.get_cov_chol()
        return cov_chol @ cov_chol.T

    def get_log_det(self):
        """Return the log determinant of log(det(sigma_p)/det(sigma_q))

        Returns:
            tensor.float: The value of the logdet
        """
        log_det_q = 2 * torch.sum(torch.log(torch.diagonal(self.get_cov_chol())))
        log_det_p = torch.sum(self.prior_log_var)
        return log_det_p - log_det_q

    def sample_ll(self, samples=1):
        cov_chol = self.get_cov_chol()
        std = torch.reshape((cov_chol @ torch.randn((self.feature_dim * self.out_dim, samples))).T, (samples, self.feature_dim, self.out_dim))
        return self.ll_mu + std

    def KL_div(self):
        log_determinant = self.get_log_det()
        trace = torch.sum(torch.divide(torch.diagonal(self.get_ll_cov()), torch.flatten(torch.exp(self.prior_log_var))))
        scalar_prod = torch.sum(torch.div(torch.square(self.prior_mu - self.ll_mu), torch.exp(self.prior_log_var)))
        kl_div = 0.5 * (log_determinant - self.ll_n_parameters + trace + scalar_prod)
        if kl_div <= 0: # sanity check
            print("kl_div", kl_div.item())
            print("logdet", log_determinant.item(),"trace", trace.item(),"scalar", scalar_prod.item())
            raise ValueError("KL div is smaller than 0")
        return kl_div
