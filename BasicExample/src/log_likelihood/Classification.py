from src.log_likelihood import LogLikelihoodMonteCarlo, LogLikelihood
import torch
import math
   

class Categorical(LogLikelihoodMonteCarlo):
    name = "Categorical"

    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, average=True):
        pred = torch.softmax(pred, dim=-1) # convert to probs
        if average:
            pred = self.average_prediction(pred)
        pred = torch.log(pred)
        torch_loss = self.nll_loss(pred, target)
        return torch_loss
        target = torch.squeeze(target)
        row_index = torch.arange(0, target.shape[0])
        smoothing = 0
        smooth_target = torch.full_like(pred, smoothing)
        smooth_target[row_index, target] = 1 - smoothing * (pred.shape[1])
        loss = - torch.mean(pred * smooth_target)
        # return loss

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

class CategoricalClosedForm(LogLikelihood):
    name = "CategoricalClosedForm"

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