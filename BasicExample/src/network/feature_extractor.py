from torch import nn, optim
import torch
from typing import List

class FC_Net(nn.Sequential):
    def __init__(self, layers: List[int], nll = nn.Tanh(), optimizer=optim.SGD, **optim_kwargs):
        layers_container = []
        for in_dim, out_dim in zip(layers[:-1], layers[1:]):
            layers_container.append(nn.Linear(in_dim, out_dim))
            layers_container.append(nll)

        super().__init__(*layers_container)
        self.optimizer: optim = optimizer(self.parameters(), **optim_kwargs)
        self.optim_args = optim_kwargs

    def delete_last_layer(self):
        new_model = nn.Sequential(*list(self.children())[:-1])
        new_model.optimizer = type(self.optimizer)(new_model.parameters(), **self.optim_args)
        return new_model

