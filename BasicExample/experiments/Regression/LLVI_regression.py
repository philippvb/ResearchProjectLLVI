import sys
import torch
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample")
from datasets.Regression.toydataset import create_dataset, sinus_mapping, dataset_to_loader, visualize_predictions
from src.network.feature_extractor import FC_Net
from src.network import LikApprox

torch.manual_seed(5)

data_noise = 0.2
x_train, y_train, x_test, y_test = create_dataset(lower=-5, upper=7, mapping=sinus_mapping,cluster_pos=[-0.5,2], data_noise=data_noise, n_datapoints=256)

lr = 1e-4
feature_extractor = FC_Net(layers=[1, 200, 100], nll = torch.nn.Tanh(),lr=lr, weight_decay=0.1)

# from src.log_likelihood.Regression import RegressionNoNoise
# lik = RegressionNoNoise()

from src.weight_distribution.Diagonal import Diagonal
from src.weight_distribution.Full import FullCovariance
# dist = Diagonal(100, 1, lr=lr)
dist = FullCovariance(100, 1, lr=lr, init_log_var=-0.5)

# from src.network import LLVINetwork

# full_net = LLVINetwork(100, 1, feature_extractor=feature_extractor, weight_dist=dist, loss_fun=lik)

# batch_size = 16
# random_permutation = torch.randperm(len(x))
# x_batch = torch.split(torch.unsqueeze(x[random_permutation], dim=1), batch_size)
# y_batch =  torch.split(torch.unsqueeze(y[random_permutation], dim=1), batch_size)

# full_net.forward_ML(x_batch[0])


# # model.load_ml_estimate("P:/Dokumente/3 Uni/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/models/Regression/Pretrained/Baseline_wdecay_e-1")
# # full_net.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# full_net.train_model(list(zip(x_batch, y_batch)), epochs=1, n_datapoints=total_points, samples=10)

from src.network.Regression import LLVIRegression
batch_size = 16
random_permutation = torch.randperm(len(x_train))
x_batch = torch.split(torch.unsqueeze(x_train[random_permutation], dim=1), batch_size)
y_batch =  torch.split(torch.unsqueeze(y_train[random_permutation], dim=1), batch_size)


net = LLVIRegression(100, 1, feature_extractor, dist, prior_log_var=-6,
tau=0.01, data_log_var=-1,#torch.log(torch.tensor([0.04])),
 lr=lr)
train_set, test_set = dataset_to_loader(x_train, y_train, x_test, y_test , batch_size=16)
# net.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# net.train_model(list(zip(x_batch, y_batch)), epochs=500, n_datapoints=512, samples=1, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)

net.train_without_VI(train_set, epochs=100)
net.train_model(train_set, epochs=500, n_datapoints=256, samples=1, method=LikApprox.CLOSEDFORM, train_hyper=True, update_freq=5)

from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
fig, (ax1, ax2) = plt.subplots(2)
visualize_predictions(net, ax1, x_train, y_train, x_test, y_test, data_noise=data_noise)
ax1.set_title("Closed form")

feature_extractor = FC_Net(layers=[1, 200, 100], nll = torch.nn.Tanh(),lr=lr, weight_decay=0.1)
net = LLVIRegression(100, 1, feature_extractor, dist, prior_log_var=-6,
tau=0.01, data_log_var=-1,#torch.log(torch.tensor([0.04])),
 lr=lr)
train_set, test_set = dataset_to_loader(x_train, y_train, x_test, y_test , batch_size=16)
# net.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# net.train_model(list(zip(x_batch, y_batch)), epochs=500, n_datapoints=512, samples=1, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)

net.train_without_VI(train_set, epochs=100)
net.train_model(train_set, epochs=1000, n_datapoints=256, samples=10, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)
visualize_predictions(net, ax2, x_train, y_train, x_test, y_test, data_noise=data_noise)
ax2.set_title("MC")

# plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Regression/cf_mc_comparison/comparison_hyperparam_opt.jpg")
plt.show()
# print(net.predict(x_batch[0]))