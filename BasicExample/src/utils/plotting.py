from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import torch

def plot_trajectories(traj, axs):
    pca = PCA(n_components=2)
    projection = pca.fit_transform(traj)
    axs.scatter(projection[:, 0], projection[:, 1], s=10)
    return projection




