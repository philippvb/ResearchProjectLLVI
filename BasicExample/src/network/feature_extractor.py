from torch import nn, optim
import torch
from typing import List
from torch.nn import functional as F

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

    @torch.no_grad()
    def add_last_layer(self, weights):
        last_layer = nn.Linear(weights.shape[0], weights.shape[1], bias=False)
        weights = torch.tensor([list(weights[:,0]), list(weights[:,1])])
        last_layer.weight.copy_(weights)
        layers = list(self.children()) + [last_layer]
        new_model = nn.Sequential(*layers)
        new_model.optimizer = type(self.optimizer)(new_model.parameters(), **self.optim_args)
        return new_model


class CNN(nn.Module):
    """Basic CNN for MNIST taken from https://nextjournal.com/gkoehler/pytorch-mnist.
    Removed the Last fully connected layer for LLVI.
    """
    def __init__(self, optimizer=optim.SGD, **optim_kwargs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.optimizer: optim = optimizer(self.parameters(), **optim_kwargs)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x



