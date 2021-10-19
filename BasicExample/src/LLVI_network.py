import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


class LLVI_network_diagonal(nn.Module):
    def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=0, lr=1e-2, tau=1) -> None:
        super(LLVI_network_diagonal, self).__init__()
        self.feature_extractor = feature_extractor

        self.ll_mu =  nn.Parameter(torch.randn(feature_dim, out_dim, requires_grad=True))
        self.ll_log_var =  nn.Parameter(torch.randn_like(self.ll_mu, requires_grad=True))
        self.prior_mu = prior_mu * torch.ones_like(self.ll_mu)
        self.prior_log_var = prior_log_var * torch.ones_like(self.ll_log_var)

        self.optimizer = optim.SGD(self.parameters(),
                        lr=lr,momentum=0.5)

        self.loss_fun = nn.NLLLoss(reduction="mean")
        self.tau = tau

    def forward(self, x, samples=1):
        features = self.feature_extractor(x)
        output = features @ self.sample_ll(samples=samples)
        likelihood = F.log_softmax(output, dim=-1) # convert to logprobs
        likelihood = torch.mean(likelihood, dim=0) # take the mean
        kl_loss = self.KL_div_gaussian_diagonal()
        return likelihood, kl_loss

    def sample_ll(self, samples=1):
        std = torch.multiply(torch.exp(0.5 * self.ll_log_var),  torch.randn((samples, ) + self.ll_log_var.size()))
        return self.ll_mu + std

    
    def KL_div_gaussian_diagonal(self):
        return 0.5 * (torch.sum(self.prior_log_var) - torch.sum(self.ll_log_var) - self.ll_mu.shape[0] + torch.sum(torch.exp(self.ll_log_var - self.prior_log_var)) + torch.sum(torch.div(torch.square(self.prior_mu - self.ll_mu), torch.exp(self.prior_log_var))))


    def train_model(self, train_loader, n_datapoints, epochs=1, samples=1):
        self.train()
        losses = []
        for epoch in range(epochs):
            kl_losses = []
            prediction_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output, kl_loss = self.forward(data, samples=samples)
                prediction_loss = self.loss_fun(output, target)
                kl_loss = + self.tau * kl_loss / n_datapoints # rescale kl_loss
                loss = prediction_loss + kl_loss
                loss.backward()
                with torch.no_grad():
                    kl_losses.append(kl_loss)
                    prediction_losses.append(prediction_loss)
                self.optimizer.step()
            epoch_loss = (sum(kl_losses) + sum(prediction_losses))/len(prediction_losses)
            print(f"Finished Epoch {epoch}\n\tmean loss {epoch_loss}\n\tmean prediction loss {sum(prediction_losses)/len(prediction_losses)}\n\tmean kl loss {sum(kl_losses)/len(kl_losses)}")

            losses.append(epoch_loss)

        return losses

    def test(self, test_loader, samples=5):
        test_losses = []
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output, kl_div = self.forward(data, samples=samples)
                test_loss += self.loss_fun(output, target).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        return test_losses

    def test_confidence(self, test_loader, ood_test_loader=None, samples=5):
        self.eval()
        confidence_batch = []
        with torch.no_grad():
            for data, target in test_loader:
                output, kl_div = self.forward(data, samples=samples)
                output_probs = F.softmax(output, dim=1)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for in distribution data is: {sum(confidence_batch)/len(confidence_batch)}")

        ood_confidence_batch = []
        with torch.no_grad():
            for data, target in ood_test_loader:
                output, kl_div = self.forward(data, samples=5)
                output_probs = F.softmax(output, dim=1)
                pred, _ = torch.max(output_probs, dim=1) # confidence in choice
                ood_confidence_batch.append(torch.mean(pred))
            print(f"The mean confidence for out-of distribution data is: {sum(ood_confidence_batch)/len(ood_confidence_batch)}")




