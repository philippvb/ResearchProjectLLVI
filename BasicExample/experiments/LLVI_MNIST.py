import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/ResearchProject/ResearchProjectLLVI')
import torch
import torchvision

batch_size_train = 32
batch_size_test = 1000
filepath = "/Users/philippvonbachmann/Documents/University/ResearchProject/ResearchProjectLLVI/BasicExample/files"
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(filepath, train=True, download=True,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
  batch_size=batch_size_train, shuffle=True)
n_datapoints = batch_size_train * len(train_loader)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(filepath, train=False, download=True,
                            transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
batch_size=batch_size_test, shuffle=True)

ood_test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(filepath, train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                torchvision.transforms.RandomHorizontalFlip(p=1)
                            ])),
batch_size=batch_size_test, shuffle=True)

def test_LLVI(network):
  network.train_model(train_loader=train_loader, n_datapoints=n_datapoints, epochs=10, samples=2)
  network.test(test_loader=test_loader)
  network.test_confidence(test_loader=test_loader, ood_test_loader=ood_test_loader)


from BasicExample.src.LLVI_network import LLVI_network_diagonal, LLVI_network_KFac, Log_likelihood_type
from BasicExample.src.basic_CNN import Net, NonBayesianNet
# test the baseline net
# non_bayes_net = NonBayesianNet()
# non_bayes_net.train_model(train_loader, epochs=10)
# non_bayes_net.test(test_loader)
# non_bayes_net.test_confidence(test_loader, ood_test_loader)


# # test the diagonal gaussian net
feature_extractor_net = Net()
Diag_net = LLVI_network_diagonal(feature_extractor=feature_extractor_net,
feature_dim=50, out_dim= 10,
prior_mu=0, prior_log_var=-2,
init_ll_mu=0,init_ll_log_var=0,
lr=1e-2, tau=10, bias=False, loss=Log_likelihood_type.CATEGORICAL)

# Diag_net.train_without_VI(train_loader, epochs=10)
# test_LLVI(Diag_net)
for data, target in test_loader:
  result = Diag_net.predict(data)
  print(result.sum(-1))

# test KFac net
# feature_extractor_net = Net()
# KFac_net = LLVI_network_KFac(feature_extractor=feature_extractor_net,
# feature_dim=50, out_dim= 10,
# A_dim=50, B_dim=10,
# prior_mu=0, prior_log_var=-2,
# init_ll_mu=0, init_ll_cov_scaling=0.1,
# lr=1e-2, tau=1)
# test_LLVI(KFac_net)


