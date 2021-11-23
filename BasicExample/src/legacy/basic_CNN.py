from sklearn.covariance import log_likelihood
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import log_, optim
from tqdm import tqdm
class Net(nn.Module):
    """Basic CNN for MNIST taken from https://nextjournal.com/gkoehler/pytorch-mnist.
    Removed the Last fully connected layer for LLVI.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return x


class NonBayesianNet(nn.Module):
    """Basic CNN for MNIST taken from https://nextjournal.com/gkoehler/pytorch-mnist.
    Used as a baseline comparison
    """
    def __init__(self, lr=1e-2):
        super(NonBayesianNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.loss_fun = nn.NLLLoss(reduction="mean")
        self.optimizer = optim.SGD(self.parameters(),
                        lr=lr,momentum=0.5)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def predict(self, x):
        return F.softmax(self.forward(x), dim=-1)

    def train_model(self, train_loader, epochs=1):
        self.train()
        losses = []
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            batch_loss = []
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.forward(data)
                log_likelihood = F.log_softmax(output, dim=1)
                loss = self.loss_fun(log_likelihood, target)
                loss.backward()
                with torch.no_grad():
                    batch_loss.append(loss)
                self.optimizer.step()
            epoch_loss = (sum(batch_loss))/len(batch_loss)
            pbar.set_description(f"Loss: {round(epoch_loss.item(), 2)}")

            losses.append(epoch_loss)
        return losses

    def test(self, test_loader):
        test_losses = []
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.forward(data)
                log_likelihood = F.log_softmax(output, dim=1)
                test_loss += self.loss_fun(log_likelihood, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        return test_losses





