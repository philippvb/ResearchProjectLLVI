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



def create_dataset_loader(lower:float, upper:float, mapping, cluster_pos:List[float], data_noise:float, n_datapoints:int, batch_size:int):
    x_train, y_train, x_test, y_test = create_dataset(lower, upper, mapping, cluster_pos, data_noise, n_datapoints)
    train_set = DataLoader(TensorDataset(torch.unsqueeze(x_train, dim=-1), torch.unsqueeze(y_train, dim=-1)), batch_size=batch_size)
    test_set = DataLoader(TensorDataset(torch.unsqueeze(x_test, dim=-1), torch.unsqueeze(y_test, dim=-1)), batch_size=batch_size)

    return train_set, test_set


def sinus_mapping(x, noise):
    return torch.sin(2.5*x) + noise * torch.randn_like(x)
