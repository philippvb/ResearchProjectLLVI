from typing import List
import torch
from torch._C import TensorType
from torch.utils.data import DataLoader, TensorDataset

def create_dataset(lower:float, upper:float, mapping, cluster_pos:List[float], data_noise:float, n_datapoints:int):
    cluster_points = n_datapoints // len(cluster_pos)
    x_train = torch.cat([mean + torch.rand(cluster_points) for mean in cluster_pos])
    y_train = mapping(x_train, noise=data_noise)
    x_test = torch.linspace(lower, upper, 100)
    y_test = mapping(x_test, noise=False)
    return x_train, y_train, x_test, y_test

def dataset_to_loader(x_train, y_train, x_test, y_test, batch_size):
    train_set = DataLoader(TensorDataset(torch.unsqueeze(x_train, dim=-1), torch.unsqueeze(y_train, dim=-1)), batch_size=batch_size)
    test_set = DataLoader(TensorDataset(torch.unsqueeze(x_test, dim=-1), torch.unsqueeze(y_test, dim=-1)), batch_size=batch_size)
    return train_set, test_set

def to_loader(x,y,batch_size):
    return DataLoader(TensorDataset(torch.unsqueeze(x, dim=-1), torch.unsqueeze(y, dim=-1)), batch_size=batch_size)



def create_dataset_loader(lower:float, upper:float, mapping, cluster_pos:List[float], data_noise:float, n_datapoints:int, batch_size:int):
    x_train, y_train, x_test, y_test = create_dataset(lower, upper, mapping, cluster_pos, data_noise, n_datapoints)
    return dataset_to_loader(x_train, y_train, x_test, y_test, batch_size)


def sinus_mapping(x, noise):
    return torch.sin(2.5*x) + noise * torch.randn_like(x)

def visualize_predictions(model, ax, x_train, y_train, x_test, y_test, noise=True, data_noise=0.1, laplace=False):
    # first add true data
    ax.plot(x_test, y_test, color="black", label="True function")
    ax.scatter(x_train, y_train, s=3, color="black")

    x_test_batch = torch.unsqueeze(x_test, dim=1) # put all data in one batch

    # predict
    if laplace:
        y_mean, y_var = model(x_test_batch)
        y_mean = torch.flatten(y_mean)
        y_std = torch.sqrt(torch.flatten(y_var) + model.sigma_noise.item()**2)
        noise_txt = f"Estimated data noise: {round(model.sigma_noise.item(), 2)}, true noise:{data_noise}"
        y_std_true_noise = torch.sqrt(torch.flatten(y_var)) + data_noise
        ax.set_xlabel(r'\begin{center}x\\*\textit{\small{' + noise_txt + r'}}\end{center}')
    else:
        with torch.no_grad():
            y_mean, y_var = model(x_test_batch)
            y_mean = torch.squeeze(y_mean)
        y_std = torch.sqrt(torch.diagonal(y_var))
        if noise:
            y_std_true_noise = y_std - torch.exp(0.5 * model.data_log_var).item() + data_noise
            noise_txt = f"Estimated data noise: {round(torch.exp(0.5 * model.data_log_var).item(), 2)}, true noise:{data_noise}"

    # visualize
    ax.plot(x_test, y_mean, color="royalblue", label="ML prediction/Mean") # mean
    # std
    ax.plot(x_test, y_mean+1.96*y_std, color="orange", label="$+-1.96 \cdot \sigma$")
    ax.plot(x_test, y_mean-1.96*y_std, color="orange")
    ax.fill_between(torch.squeeze(x_test),y_mean+1.96*y_std,y_mean-1.96*y_std, alpha=0.1, color="orange")
    
    # true noise
    if noise:
        ax.plot(x_test, y_mean+1.96*y_std_true_noise, color="darkorange", label="$\sigma$ with true noise", linestyle="dashed")
        ax.plot(x_test, y_mean-1.96*y_std_true_noise, color="darkorange",  linestyle="dashed")
        ax.set_xlabel(r'\begin{center}x\\*\textit{\small{' + noise_txt + r'}}\end{center}')
    else:
        ax.set_label("x")


    # labels
    ax.set_ylim(torch.min(y_test) - 2, torch.max(y_test) + 2)
    # ax.set_title(model.model_config["kernel_name"])
    ax.set_ylabel("y")

@torch.no_grad()
def visualize_features(model, ax, x_test):
    features = model.feature_extractor(torch.unsqueeze(x_test, dim=1)).detach()
    n_features = features.shape[1]
    for i in range(n_features):
        ax.plot(x_test, features[:, i] * model.weight_distribution.mu[i], alpha=0.1)