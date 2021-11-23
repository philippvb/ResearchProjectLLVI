from src.log_likelihood import LogLikelihood, LogLikelihoodMonteCarlo
import torch
import math

class Regression(LogLikelihoodMonteCarlo):
    name = 'Regression'

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, data_var:torch.Tensor, average=True):
        """Returns the mean negative log likelihood of the target (true) data given
        the prediction in the gaussian case/regression. Scaled by 1/datapoints.

        Args:
            pred (torch.Tensor): the prediction
            target (torch.Tensor): the target data
            mean (bool, optional): If true, assumes multiple predictions in dim 0 and averages over them. Defaults to True.

        Returns:
            torch.Tensor: The negative log-likelihood
        """
        if average:
            pred = self.average_prediction(pred)
        squared_diff = torch.mean(torch.square(pred - target))
        return 0.5 * (math.log(2 * math.pi * data_var) + squared_diff / data_var)
        

class RegressionNoNoise(LogLikelihoodMonteCarlo):
    name = 'RegressionNoNoise'
    torch_loss =  torch.nn.MSELoss()

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, average=True):
        """Returns the mean negative log likelihood of the target (true) data given
        the prediction in the gaussian case/regression. Scaled by 1/datapoints.

        Args:
            pred (torch.Tensor): the prediction
            target (torch.Tensor): the target data
            mean (bool, optional): If true, assumes multiple predictions in dim 0 and averages over them. Defaults to True.

        Returns:
            torch.Tensor: The negative log-likelihood
        """
        if average:
            pred = self.average_prediction(pred)
        return self.torch_loss(pred, target)

class ClosedFormRegression(LogLikelihood):
    name =  "ClosedFormRegression"
    scalar = math.log(2 * math.pi)

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred_mean: torch.Tensor,pred_cov: torch.Tensor, target: torch.Tensor, data_var: torch.Tensor):
        """        
        Closed form expression for expectation under the parameter distribution
        of data log likelihood for gaussian likelihood,
        E_N(theta, mu, cov)[log(N(features @ theta, y, data_var))]

        Args:
        features (torch.Tensor): features of size batch_size x n_features
        target (torch.Tensor): target of size batch_size x 1
        mu (torch.Tensor): mean of the last-layer weights theta
        cov (torch.Tensor): covariance of the last layer weights theta
        data_var (torch.Tensor): variance of the data for the data likelihood

        Returns:
        torch.Tensor: The expected data log likelihood under the model
        """
        batch_size = pred_mean.shape[0]
        data_log_var = torch.log(data_var)
        trace = torch.trace(pred_cov)
        square_terms = torch.sum(torch.square(pred_mean)) + torch.sum(torch.square(target))
        error = - 2 * torch.sum(pred_mean * target)
        # return 0.5 * (ClosedFormRegression.scalar +  data_log_var + (trace + square_terms + error) / data_var / batch_size)
        return data_log_var + (trace + square_terms + error) / data_var / batch_size


