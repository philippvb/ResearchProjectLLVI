import sys
sys.path.append('P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI')

from sklearn import datasets
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
from matplotlib import pyplot as plt
import matplotlib

n_datapoints=1024
batch_size = 32
epochs=50

x_np,y_np = datasets.make_moons(n_samples=n_datapoints, shuffle=True, noise=0.2, random_state=1234)



x = torch.split(torch.from_numpy(x_np.astype(np.float32)), batch_size) 
y = torch.split(torch.from_numpy(y_np), batch_size)





class FC_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

from BasicExample.src.LLVI_network import LLVI_network_diagonal

feature_extractor = FC_Net()

LLVI_diag = LLVI_network_diagonal(feature_extractor=feature_extractor,
feature_dim=20, out_dim=2,
prior_mu=0, prior_log_var=-3,
init_ll_mu=0, init_ll_log_var=0, tau=1)

LLVI_diag.train_model(list(zip(x,y)), n_datapoints=n_datapoints, epochs=epochs, samples=5)



plt.figure()
plt.xlabel('x1')
plt.ylabel('x2')
test_rng = np.linspace(-2, 3, 50)
X1_test, X2_test = np.meshgrid(test_rng, test_rng)
X_test = torch.tensor(np.stack([X1_test.ravel(), X2_test.ravel()]).T, dtype=torch.float)
print(X_test[:10])

with torch.no_grad():
    log_lik, _ = LLVI_diag(X_test, samples=20)
    map_conf = torch.exp(log_lik).max(1).values.reshape(50, 50)
    plt.contourf(X1_test, X2_test, map_conf, cmap='binary')
    cbar = plt.colorbar(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cbar.set_label('confidence')


plt.scatter(x_np[:, 0], x_np[:, 1], c=y_np, cmap=matplotlib.colors.ListedColormap(["red", "blue"]))
plt.show()






