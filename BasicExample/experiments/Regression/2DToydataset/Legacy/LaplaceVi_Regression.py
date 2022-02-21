from multiprocessing.sharedctypes import Value
import sys
import torch
sys.path.append("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample")
from src.log_likelihood import LogLikelihoodMonteCarlo
from datasets.Regression.toydataset import create_dataset, sinus_mapping, dataset_to_loader, visualize_predictions, to_loader
from src.network.feature_extractor import FC_Net
from src.network import LikApprox

torch.manual_seed(1)

data_noise = 0.2
x_train, y_train, x_test, y_test = create_dataset(lower=-5, upper=7, mapping=sinus_mapping,cluster_pos=[-0.5, 2], data_noise=data_noise, n_datapoints=256)

lr = 1e-4
feature_extractor = FC_Net(layers=[1, 200, 100], nll = torch.nn.Tanh(),lr=lr, weight_decay=0.1)

from src.network.Regression import LLVIRegression
batch_size = 16
random_permutation = torch.randperm(len(x_train))
x_batch = torch.split(torch.unsqueeze(x_train[random_permutation], dim=1), batch_size)
y_batch =  torch.split(torch.unsqueeze(y_train[random_permutation], dim=1), batch_size)
train_set, test_set = dataset_to_loader(x_train, y_train, x_test, y_test , batch_size=16)

from src.laplace import LaplaceVI
# from src.laplace.custom_hessian import LaplaceVI
net = LaplaceVI(100, 1, feature_extractor, LogLikelihoodMonteCarlo, prior_log_var=0)

train_set_2 = to_loader(x_train, y_train, batch_size=256)
# net.set_cov_backpack(train_set)
# net.train_without_VI(train_set, epochs=200)
# net.train_model(train_set, epochs=200, n_datapoints=512, samples=1, method=LikApprox.MONTECARLO, train_hyper=False, update_freq=5)
# net.set_cov(train_set)
# print(net.cov_approx)

# for data, target in train_set:
#     features = net.feature_extractor(data)
#     print(net.get_cov_ggn(features, target)[:10])
#     print(net.get_cov_ggn_2(features, target)[:10])
#     raise ValueError

# net(torch.unsqueeze(x_train, dim=-1))
net.train_without_VI(train_set, epochs=100)
net.set_cov(train_set)
# net.train_model(train_set, epochs=500, n_datapoints=256, samples=1, method=LikApprox.CLOSEDFORM, train_hyper=True, update_freq=5)

from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.size": 14,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
fig, (ax1, ax2) = plt.subplots(2)
visualize_predictions(net, ax1, x_train, y_train, x_test, y_test, data_noise=data_noise, noise=False)
ax1.set_title("Closed form")

net.cov_approx = net.compute_cov_by_hand(train_set)
visualize_predictions(net, ax2, x_train, y_train, x_test, y_test, data_noise=data_noise, noise=False)
ax2.set_title("new form")
# feature_extractor = FC_Net(layers=[1, 200, 100], nll = torch.nn.Tanh(),lr=lr, weight_decay=0.1)
# net = LLVIRegression(100, 1, feature_extractor, dist, prior_log_var=-6,
# tau=0.01, data_log_var=-1,#torch.log(torch.tensor([0.04])),
#  lr=lr)
# train_set, test_set = dataset_to_loader(x_train, y_train, x_test, y_test , batch_size=16)
# # net.train_without_VI(list(zip(x_batch, y_batch)), epochs=100)
# # net.train_model(list(zip(x_batch, y_batch)), epochs=500, n_datapoints=512, samples=1, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)

# # net.train_without_VI(train_set, epochs=100)
# # net.train_model(train_set, epochs=1000, n_datapoints=256, samples=10, method=LikApprox.MONTECARLO, train_hyper=True, update_freq=5)
# visualize_predictions(net, ax2, x_train, y_train, x_test, y_test, data_noise=data_noise, noise=False)
# ax2.set_title("MC")

# # plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/ResearchProject/ResearchProjectLLVI/BasicExample/results/Regression/cf_mc_comparison/comparison_hyperparam_opt.jpg")
plt.show()
# # print(net.predict(x_batch[0]))