from abc import ABC, abstractmethod
from turtle import pen
from src.log_likelihood import LogLikelihoodMonteCarlo, LogLikelihood
import torch
import math
   

class Categorical(LogLikelihoodMonteCarlo):
    name = "Categorical"

    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, average=True):
        # pred = torch.log_softmax(pred, dim=-1) # convert to probs
        if average:
            pred = torch.softmax(pred, dim=-1)
            # target = torch.squeeze(target)
            # row_index = torch.arange(0, target.shape[0])
            # target_full_tensor = torch.full_like(pred, 0)
            # target_full_tensor[:,row_index, target] = 1
            # loss = - torch.mean(pred * target_full_tensor)
            # return loss

            pred = self.average_prediction(pred)
            pred = torch.log(pred)
            torch_loss = self.nll_loss(pred, target)
            return torch_loss
        else:
            pred = torch.log_softmax(pred, dim=-1)
            return self.nll_loss(pred, target)

class Categorical2Classes(LogLikelihoodMonteCarlo):
    name = "Categorical2Classes"

    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, average=True):
        prob_class_0 = torch.sigmoid(pred)
        if average:
            pred = self.average_prediction(pred)
        probs = torch.log(torch.hstack((prob_class_0, 1-prob_class_0)))
        nll_pyt = self.nll_loss(probs, target)
        return nll_pyt

class CategoricalProbitApprox(LogLikelihood):
    name = "CategoricalClosedFormSigmoidApproximation"

    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, pred_mean:torch.Tensor, pred_cov:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        pred_var = torch.diagonal(pred_cov, dim1=1, dim2=2)
        prediction = torch.log_softmax(self.probit_approx(pred_mean, pred_var), dim=-1)
        return self.nll_loss(prediction, target)

    def probit_approx(self, mean, cov):
        """Probit approximation given a mean and a covariance
        """
        return mean / torch.sqrt(1 + math.pi * cov / 8)

    def cf_log_lik(self, mean, cov):
        return torch.log(self.probit_approx(mean, cov))


class CategoricalClosedFormLSEApproximation(LogLikelihood, ABC):
    name = "CategoricalClosedFormApproximation"

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred_mean:torch.Tensor, pred_cov:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        left_hand = pred_mean[torch.arange(target.shape[0]), torch.squeeze(target)] # same for all approximations since closed form
        lse = self._get_lse_expectation(pred_mean, pred_cov, target)
        loss = - torch.mean(left_hand - lse) # take mean over batch and negative for log-lik
        return loss

    @abstractmethod
    def _get_lse_expectation(self, pred_mean:torch.Tensor, pred_cov:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        """Returns the mean of the log-sum-exponential(lse) under a Gaussian distribution

        Args:
            pred_mean (torch.Tensor): The mean of the prediction, size batch x n_classes
            pred_cov (torch.Tensor): The covariance of the prediction, size batch x n_classes x n_classes
            target (torch.Tensor): The true class one-hot encoded, size batch x 1

        Returns:
            torch.Tensor: lse, size batch x 1
        """
        raise NotImplementedError

class CategoricalJennsenApprox(CategoricalClosedFormLSEApproximation):
    """Lower bound on the categorical log-likelihood based on the Jennsen
    inequality taken from:
    http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf, section 21.8.4.2
    """
    name = "CategoricalClosedFormJennsenApproximation"

    def __init__(self) -> None:
        super().__init__()

    def _get_lse_expectation(self, pred_mean: torch.Tensor, pred_cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        inner = torch.exp(pred_mean + torch.diagonal(pred_cov, dim1=1, dim2=2) / 2)
        sum = torch.sum(inner, dim=1)
        return torch.log(sum)



class CategoricalBohningApprox(CategoricalClosedFormLSEApproximation):
    """Lower bound on the categorical log-likelihood based on the Bohning
    bound taken from:
    http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf, section 21.8.2
    """
    name = "CategoricalClosedFormBohningApproximation"

    def __init__(self) -> None:
        super().__init__()

    def lse(self, x):
        return torch.log(torch.sum(torch.exp(x),dim=-1))

    def _get_lse_expectation(self, pred_mean: torch.Tensor, pred_cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred_mean.shape[-1]
        # required precomputations
        a = (torch.eye(n_classes) - torch.full((n_classes, n_classes), 1)/n_classes)/2
        g = torch.softmax(pred_mean, dim=-1)
        b = torch.einsum("ac,bc -> ba", a, pred_mean) - g
        a_mean_scalar_prod = torch.diagonal(pred_mean @ a @ pred_mean.T) # batch product of pred_mean @ A  @ pred_mean.T
        # a_mean_scalar_prod = torch.einsum("ab, bc, cd -> ad", pred_mean, a, pred_mean.T)
        c = a_mean_scalar_prod/2 - torch.einsum("bc,bc -> b", g, pred_mean) + self.lse(pred_mean)
        trace = torch.sum(torch.diagonal(torch.einsum("ac,bcd->bad", a, pred_cov), dim1=1, dim2=2), dim=-1) # trace of A @ pred_cov
        # final form
        lse =  trace/2 + a_mean_scalar_prod /2 - torch.einsum("bc, bc -> b", b, pred_mean) + c
        return lse



class CategoricalMultiDeltaApprox(CategoricalClosedFormLSEApproximation):
    """Lower bound on the categorical log-likelihood based on the Multivariate Delta Method taken from:
    http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf, section 21.8.4.3
    """
    name = "CategoricalClosedFormMultivariateDeltaApproximation" 

    def __init__(self) -> None:
        super().__init__()

    def batch_outer(self, x):
        return torch.einsum('bp, br->bpr', x,x)

    def lse(self, x):
        return torch.log(torch.sum(torch.exp(x),dim=-1))

    def batch_diagonal(self, x):
        id_tensor = torch.eye(x.shape[1])
        return torch.einsum("ab, bc -> abc", x, id_tensor)

    def _get_lse_expectation(self, pred_mean: torch.Tensor, pred_cov: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lse = self.lse(pred_mean)
        g = torch.softmax(pred_mean, dim=-1)
        h = self.batch_diagonal(g) + self.batch_outer(g)
        trace_scalar = torch.einsum("ab, abc, ca -> a", pred_mean, h, pred_mean.T)
        trace = torch.sum(torch.diagonal(pred_cov, dim1=1, dim2=2), dim=-1)
        return lse + trace_scalar * trace / 2


        




