
from sklearn import datasets
import torch
import numpy as np

def create_train_set(n_datapoints, noise=0.2):
    x_np,y_np = datasets.make_moons(n_samples=n_datapoints, shuffle=True, noise=noise, random_state=1234)
    x = torch.from_numpy(x_np.astype(np.float32))
    y = torch.from_numpy(y_np)
    return x, y

def create_test_points(lower, upper, n_datapoints):
    test_rng = np.linspace(lower, upper, n_datapoints)
    X1_test, X2_test = np.meshgrid(test_rng, test_rng)
    X_test = torch.tensor(np.stack([X1_test.ravel(), X2_test.ravel()]).T, dtype=torch.float)
    return X_test, X1_test, X2_test



