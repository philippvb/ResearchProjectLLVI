from syndata.synthetic_datasets import generate_p2
import torch
from matplotlib import pyplot as plt
import matplotlib

n_datapoints = 200
class_datapoints = int(n_datapoints/2)
X,y = generate_p2([class_datapoints]*2)
X = torch.tensor(X)
y = torch.tensor(y)

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=matplotlib.colors.ListedColormap(["red", "blue"]), s=10)
# plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Classification/init_laplace/ll_train_only.jpg")
plt.show()

