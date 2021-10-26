import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class LLVI_network(nn.Module):
    """Base class for Last-Layer Variational Inference Networks (LLVI), categorical networks.
    """
    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, lr=1e-2, tau=1) -> None:
        super(LLVI_network, self).__init__()
        self.feature_extractor = feature_extractor
        self.loss_fun = nn.NLLLoss(reduction="mean")
        self.tau = tau

        self.prior_mu = torch.full((feature_dim, out_dim), fill_value=prior_mu, requires_grad=True, dtype=torch.float32)
        self.prior_log_var = torch.full((feature_dim, out_dim), fill_value=prior_log_var, requires_grad=True, dtype=torch.float32)
        self.prior_optimizer = optim.SGD([self.prior_mu, self.prior_log_var], lr=lr, momentum=0.8) # optimizer for prior

    def forward(self, x, samples=1):
        features = self.feature_extractor(x)
        output = features @ self.sample_ll(samples=samples)
        log_likelihood = F.log_softmax(output, dim=-1) # convert to logprobs
        log_likelihood = torch.mean(log_likelihood, dim=0) # take the mean
        kl_loss = self.KL_div()
        return log_likelihood, kl_loss

    def sample_ll(self, samples=1):
        raise NotImplementedError

    
    def KL_div(self):
        raise NotImplementedError


    def train_model(self, train_loader, n_datapoints, epochs=1, samples=1, train_hyper=False, update_freq=10):
        self.train()
        epoch_losses = []
        for epoch in range(epochs):
            batch_kl_losses = []
            batch_prediction_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                log_likelihood, kl_loss = self.forward(data, samples=samples)
                prediction_loss = self.loss_fun(log_likelihood, target)
                kl_loss = self.tau * kl_loss / n_datapoints # rescale kl_loss
                loss = prediction_loss + kl_loss
                loss.backward()
                with torch.no_grad():
                    batch_kl_losses.append(kl_loss)
                    batch_prediction_losses.append(prediction_loss)
                self.optimizer.step()

            # update the hyperparameters
            if train_hyper and (epoch % update_freq) == 0:
                self.train_hyper(train_loader, n_datapoints, samples)

            current_epoch_loss = (sum(batch_kl_losses) + sum(batch_prediction_losses))/len(batch_prediction_losses)
            print(f"Finished Epoch {epoch}\n\tmean loss {current_epoch_loss}\n\tmean prediction loss {sum(batch_prediction_losses)/len(batch_prediction_losses)}\n\tmean kl loss {sum(batch_kl_losses)/len(batch_kl_losses)}")

            epoch_losses.append(current_epoch_loss)
        return epoch_losses


    def train_hyper(self, train_loader, n_datapoints, samples):
        for batch_idx, (data, target) in enumerate(train_loader):
            self.prior_optimizer.zero_grad()
            log_likelihood, kl_loss = self.forward(data, samples=samples)
            prediction_loss = self.loss_fun(log_likelihood, target)
            kl_loss = self.tau * kl_loss / n_datapoints # rescale kl_loss
            loss = prediction_loss + kl_loss
            loss.backward()
            self.prior_optimizer.step()

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
                log_likelihood, kl_div = self.forward(data, samples=samples)
                output_probs = torch.exp(log_likelihood)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for in distribution data is: {sum(confidence_batch)/len(confidence_batch)}")

        ood_confidence_batch = []
        with torch.no_grad():
            for data, target in ood_test_loader:
                log_likelihood, kl_div = self.forward(data, samples=5)
                output_probs = torch.exp(log_likelihood)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                ood_confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for out-of distribution data is: {sum(ood_confidence_batch)/len(ood_confidence_batch)}")

class LLVI_network_diagonal(LLVI_network):

    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_log_var=0, lr=1e-2, tau=1) -> None:
        super(LLVI_network_diagonal, self).__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau)
        
        self.ll_mu = nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim), requires_grad=True)
        self.ll_log_var = nn.Parameter(init_ll_log_var + torch.randn_like(self.ll_mu), requires_grad=True)
        self.optimizer = optim.SGD(
            # self.parameters(),
            [{'params': self.feature_extractor.parameters(), "weight_decay": 0.1}, {"params": [self.ll_mu, self.ll_log_var]}],
            lr=lr,momentum=0.8)

    def sample_ll(self, samples=1):
        std = torch.multiply(torch.exp(0.5 * self.ll_log_var),  torch.randn((samples, ) + self.ll_log_var.size()))
        return self.ll_mu + std

    def KL_div(self):
        return 0.5 * (torch.sum(self.prior_log_var) - torch.sum(self.ll_log_var) - self.ll_mu.shape[0] + torch.sum(torch.exp(self.ll_log_var - self.prior_log_var)) + torch.sum(torch.div(torch.square(self.prior_mu - self.ll_mu), torch.exp(self.prior_log_var))))


class LLVI_network_KFac(LLVI_network):
    def __init__(self, feature_extractor, feature_dim, out_dim, A_dim, B_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_cov_scaling=1, lr=1e-2, tau=1) -> None:
        super(LLVI_network_KFac, self).__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau)

        # feature dimensions
        self.feature_dim = feature_dim
        self.out_dim = out_dim

        # last-layer mean 
        self.ll_mu =  nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim, requires_grad=True))
        # cholesky decomposition of factors
        self.chol_a_lower = nn.Parameter(init_ll_cov_scaling * torch.randn((A_dim, A_dim), requires_grad=True)) # lower triangular matrix without diagonal
        self.chol_a_log_diag =  nn.Parameter(init_ll_cov_scaling * torch.randn(A_dim, requires_grad=True)) # separate diagonal (log since it has to be positive)
        self.chol_b_lower = nn.Parameter(init_ll_cov_scaling * torch.randn((B_dim, B_dim), requires_grad=True))
        self.chol_b_log_diag = nn.Parameter(init_ll_cov_scaling * torch.randn(B_dim, requires_grad=True))
        self.optimizer = optim.SGD(self.parameters(),
        lr=lr,momentum=0.5) # init optimizer here in oder to get all the parameters in


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
        kl_div = 0.5 * (log_determinant - self.ll_mu.shape[0] + trace + scalar_prod)
        if kl_div <= 0: # sanity check
            print("kl_div", kl_div.item())
            print("logdet", log_determinant.item(),"trace", trace.item(),"scalar", scalar_prod.item())
            raise ValueError("KL div is smaller than 0")
        return kl_div




class LLVI_network_full_Cov(LLVI_network):
    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, init_ll_mu=0, init_ll_log_var=0, init_ll_cov_scaling=1, lr=1e-2, tau=1) -> None:
        super().__init__(feature_extractor, feature_dim, out_dim, prior_mu=prior_mu, prior_log_var=prior_log_var, lr=lr, tau=tau)

        # last-layer mean 
        self.ll_mu =  nn.Parameter(init_ll_mu + torch.randn(feature_dim, out_dim, requires_grad=True))
        # ll covariance
        cov_dim = feature_dim*out_dim
        self.cov_lower = nn.Parameter(init_ll_cov_scaling * (torch.randn((cov_dim, cov_dim), requires_grad=True))) # lower triangular matrix without diagonal
        self.cov_log_diag =  nn.Parameter(init_ll_log_var + torch.randn(cov_dim, requires_grad=True)) # separate diagonal (log since it has to be positive)
        self.optimizer = optim.SGD(self.parameters(),
                lr=lr,momentum=0.8) # init optimizer here in oder to get all the parameters in

    def get_cov_chol(self):
        cov_lower_diag = torch.tril(self.chol_a_lower, diagonal=-1)
        cov = cov_lower_diag + torch.diag(torch.exp(self.chol_a_log_diag))
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
        kl_div = 0.5 * (log_determinant - self.ll_mu.shape[0] + trace + scalar_prod)
        if kl_div <= 0: # sanity check
            print("kl_div", kl_div.item())
            print("logdet", log_determinant.item(),"trace", trace.item(),"scalar", scalar_prod.item())
            raise ValueError("KL div is smaller than 0")
        return kl_div
