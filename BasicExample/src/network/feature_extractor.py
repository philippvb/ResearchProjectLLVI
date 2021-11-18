from torch import nn, optim
from abc import ABC, abstractmethod
import torch

class FeatureExtractor(nn.Module, ABC):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor):
        raise NotImplementedError



class FC_Net(FeatureExtractor):
    def __init__(self, nll = nn.Tanh(), optimizer=optim.SGD, **optim_kwargs):
        super().__init__()
        self.fc1 = nn.Linear(1, 200)
        self.fc2 = nn.Linear(200, 100)
        self.nll = nn.Tanh()
        self.optimizer = optimizer(self.parameters(), **optim_kwargs)

    def forward(self, x):
        h1 = self.nll(self.fc1(x))
        h2 = self.nll(self.fc2(h1))
        return h2