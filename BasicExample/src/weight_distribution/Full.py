from src.weight_distribution import WeightDistribution
import torch
from torch import nn, optim

class FullCovariance(WeightDistribution):

    def __init__(self, feature_dim: int, out_dim: int, init_mu=0, init_log_var=0, init_cov_scaling:int=0.1, optimizer=optim.SGD, **optim_kwargs) -> None:
        super().__init__(feature_dim, out_dim)
        self.mu = nn.Parameter(init_mu + torch.randn(feature_dim, out_dim), requires_grad=True)
        self.cov_lower = nn.Parameter(init_cov_scaling * (torch.randn((self.n_parameters, self.n_parameters), requires_grad=True))) # lower triangular matrix without diagonal
        self.cov_log_diag =  nn.Parameter(init_log_var + torch.randn(self.n_parameters, requires_grad=True)) # separate diagonal (log since it has to be positive)
        self.optimizer = optimizer([self.mu, self.cov_lower, self.cov_log_diag], **optim_kwargs) # init optimizer here in oder to get all the parameters in
        

    def get_mu(self) -> torch.Tensor:
        return self.mu

    def get_cov_chol(self) -> torch.Tensor:
        cov_lower_diag = torch.tril(self.cov_lower, diagonal=-1)
        cov = cov_lower_diag + torch.diag(torch.exp(self.cov_log_diag))
        return cov

    def sample(self, samples=10):
        cov_chol = self.get_cov_chol()
        std = torch.reshape((cov_chol @ torch.randn((self.feature_dim * self.out_dim, samples))).T, (samples, self.feature_dim, self.out_dim))
        return self.mu + std

    def get_log_det(self, prior_log_var):
        """Return the log determinant of log(det(sigma_p)/det(sigma_q))

        Returns:
            tensor.float: The value of the logdet
        """
        log_det_q = 2 * torch.sum(torch.log(torch.diagonal(self.get_cov_chol())))
        log_det_p = torch.sum(prior_log_var)
        return log_det_p - log_det_q

    def KL_div(self, prior_mu, prior_log_var) -> torch.Tensor:
        log_determinant = self.get_log_det(prior_log_var)
        trace = torch.sum(torch.divide(torch.diagonal(self.get_cov()), torch.flatten(torch.exp(prior_log_var))))
        scalar_prod = torch.sum(torch.div(torch.square(prior_mu - self.mu), torch.exp(prior_log_var)))
        kl_div = 0.5 * (log_determinant - self.n_parameters + trace + scalar_prod)
        if kl_div <= 0: # sanity check
            print("kl_div", kl_div.item())
            print("logdet", log_determinant.item(),"trace", trace.item(),"scalar", scalar_prod.item())
            raise ValueError("KL div is smaller than 0")
        return kl_div

    def update_cov(self, new_cov:torch.Tensor) -> None:
        new_cov = new_cov.detach().clone() # make sure we create new matrix
        cholesky = torch.linalg.cholesky(new_cov)
        log_variance = torch.log(torch.diag(cholesky))
        cov_lower = torch.tril(cholesky, diagonal=-1)
        self.cov_lower = nn.Parameter(cov_lower, requires_grad=True)
        self.cov_log_diag =  nn.Parameter(log_variance, requires_grad=True)
        self.optimizer.param_groups[0]["params"] = [self.mu, self.cov_lower, self.cov_log_diag] # update parameters

