from abc import ABC, abstractmethod, abstractproperty
import torch


class LogLikelihood(ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self) -> torch.Tensor:
        raise NotImplementedError()


class LogLikelihoodMonteCarlo(LogLikelihood):

    def __init__(self) -> None:
        super().__init__()

    def average_prediction(self, prediction:torch.Tensor) -> torch.Tensor:
        return torch.mean(prediction, dim=0)

