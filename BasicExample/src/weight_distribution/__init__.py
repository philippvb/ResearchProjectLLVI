from abc import ABC, abstractmethod
import torch

class WeightDistribution(ABC):
    def __init__(self, feature_dim:int, out_dim:int) -> None:
        super().__init__()
        self.feature_dim=feature_dim
        self.out_dim = out_dim

    
    # ll distribution
    def sample(self, samples=10):
        cov_chol = self.get_cov_chol()
        std = torch.reshape((cov_chol @ torch.randn((self.feature_dim * self.out_dim, samples))).T, (samples, self.feature_dim, self.out_dim))
        return self.ll_mu + std

    @abstractmethod
    def get_mu(self) -> torch.Tensor:
        raise NotImplementedError

    # @abstractmethod
    def get_cov_chol(self) -> torch.Tensor:
        raise NotImplementedError

    def get_cov(self) -> torch.Tensor:
        """Create the covariance matrix

        Returns:
            torch.tensor: The covariance matrix
        """
        chol_fac =  self.get_cov_chol()
        return chol_fac @ chol_fac.T

    @abstractmethod
    def KL_div(self, prior_mu, prior_log_var) -> torch.Tensor:
        """Compute the Kl divergence of the parameter distribution wrt. to a diagonal Gaussian Prior

        Args:
            prior_mu ([type]): [description]
            prior_log_var ([type]): [description]

        Returns:
            torch.Tensor: [description]
        """
        raise NotImplementedError