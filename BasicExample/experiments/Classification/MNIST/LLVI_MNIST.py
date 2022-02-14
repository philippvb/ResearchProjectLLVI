import sys
sys.path.append('/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample')
import torch
import torchvision

from src.network.feature_extractor import CNN
from src.network.Classification import LLVIClassification
from src.network import LikApprox, PredictApprox
from src.weight_distribution.Full import FullCovariance
from src.weight_distribution.Diagonal import Diagonal
from src.network import LLVINetwork

batch_size_train = 32
batch_size_test = 1000
filepath = "/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/datasets/Classification"
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(filepath, train=True, download=False,
                            transform=torchvision.transforms.Compose([
                              torchvision.transforms.ToTensor(),
                              torchvision.transforms.Normalize(
                                (0.1307,), (0.3081,))
                            ])),
  batch_size=batch_size_train, shuffle=True)
n_datapoints = batch_size_train * len(train_loader)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(filepath, train=False, download=False,
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

def test_confidence(model:LLVINetwork, test_loader, ood_test_loader=None, samples=5):
    model.eval()
    confidence_batch = []
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data, method=PredictApprox.MONTECARLO ,samples=samples)
            pred, _ = torch.max(output, dim=1) # confidence in choice
            confidence_batch.append(torch.mean(pred))
        print(f"The mean confidence for in distribution data is: {sum(confidence_batch)/len(confidence_batch)}")

    ood_confidence_batch = []
    with torch.no_grad():
        for data, target in ood_test_loader:
            output = model(data, method=PredictApprox.MONTECARLO ,samples=samples)
            pred, _ = torch.max(output, dim=1) # confidence in choice
            ood_confidence_batch.append(torch.mean(pred))
        print(f"The mean confidence for out-of distribution data is: {sum(ood_confidence_batch)/len(ood_confidence_batch)}")


lr = 1e-4
feature_extractor = CNN(optimizer=torch.optim.Adam, lr=lr)
dist = Diagonal(50, 10, lr=lr)
net = LLVIClassification(50, 10, feature_extractor, dist,
prior_log_var=-1, optimizer_type=torch.optim.Adam,
tau=1, lr=lr)

# net.train_without_VI(train_loader, epochs=2)
net.train_model(train_loader, epochs=10, samples=10, n_datapoints=n_datapoints)
test_confidence(net, test_loader, ood_test_loader, samples=10)
