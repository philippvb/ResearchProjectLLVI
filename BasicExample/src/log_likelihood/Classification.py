from src.log_likelihood import LogLikelihoodMonteCarlo
import torch   
   

class Categorical(LogLikelihoodMonteCarlo):
    name = "Categorical"

    def __init__(self) -> None:
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def __call__(self, pred:torch.Tensor, target:torch.Tensor, average=True):
        pred = torch.log_softmax(pred, dim=-1) # convert to logprobs
        if average:
            pred = self.average_prediction(pred)
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