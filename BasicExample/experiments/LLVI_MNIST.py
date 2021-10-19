import sys
sys.path.append('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI')

import torch
import torchvision

batch_size_train = 32
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/files', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)
n_datapoints = batch_size_train * len(train_loader)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/files', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
batch_size=batch_size_test, shuffle=True)

ood_test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/files', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                torchvision.transforms.RandomHorizontalFlip(p=0.5)
                            ])),
batch_size=batch_size_test, shuffle=True)

from BasicExample.src.LLVI_network import LLVI_network_diagonal
from BasicExample.src.basic_CNN import Net

for tau in [0, 1, 10]:
    feature_extractor_net = Net()
    LLVI_net = LLVI_network_diagonal(feature_extractor=feature_extractor_net, feature_dim=50,
    out_dim= 10, prior_mu=0, prior_log_var=0, lr=1e-2, tau=tau)
    LLVI_net.train_model(train_loader=train_loader, n_datapoints=n_datapoints, epochs=10, samples=1)
    LLVI_net.test(test_loader=test_loader)
    LLVI_net.test_confidence(test_loader=test_loader, ood_test_loader=ood_test_loader)
