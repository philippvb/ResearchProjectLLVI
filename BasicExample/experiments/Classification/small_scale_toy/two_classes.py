import torch
from torch import nn
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
import asdfghjkl

mu1 = torch.tensor([-5, 0], dtype=torch.float)
mu2 = torch.tensor([2,5], dtype=torch.float)
var = 1.5*torch.eye(2, dtype=torch.float)
dist1 = torch.distributions.MultivariateNormal(mu1, var)
dist2 = torch.distributions.MultivariateNormal(mu2, var)

n_datapoints = 400
data1 = dist1.sample((int(n_datapoints/2),))
data2 = dist2.sample((int(n_datapoints/2),))
train_data = torch.cat((data1, data2), dim=0)
train_target = torch.tensor([0] * int(n_datapoints/2) + [1] * int(n_datapoints/2), dtype=torch.float).unsqueeze(dim=1)
# train_set = train_set[torch.randperm(train_set.shape[0]), :]

fig, axs = plt.subplots(2,2)
axs = axs.flat
axs[0].scatter(data1[:,0], data1[:,1], s=1)
axs[0].scatter(data2[:,0], data2[:,1], s=1)
axs[0].set_title("Data")
# plt.show()

# define model
map_model = nn.Sequential(nn.Linear(2,1, bias=False))
map_tracking = map_model[0].weight.detach().clone()
optimizer = torch.optim.SGD(map_model.parameters(), lr=1e-3)
loss_fun = nn.BCELoss()

weight_prior_mean = 0
weight_prior_variance = 2
weight_prior = torch.distributions.MultivariateNormal(torch.full((1,2), weight_prior_mean), weight_prior_variance * torch.eye(2))

epochs = 10000
for i in range(epochs):
    optimizer.zero_grad()
    pred = torch.sigmoid(map_model(train_data))
    # print(pred, train_target)
    log_lik = loss_fun(pred, train_target)
    log_prior = - weight_prior.log_prob(map_model[0].weight)
    loss = log_prior + log_lik
    loss.backward()
    optimizer.step()
    map_tracking = torch.cat([map_tracking, map_model[0].weight.detach().clone()], dim=0)
print(log_lik)


# visualize predictions
space_x = torch.linspace(train_data[:,0].min(), train_data[:,0].max(), 100)
space_y = torch.linspace(train_data[:,1].min(), train_data[:,1].max(), 100)
gridx, grid_y = torch.meshgrid(space_x, space_y)
X_test = torch.stack([gridx.ravel(), grid_y.ravel()]).T
with torch.no_grad():
    pred = map_model(X_test)
    cax1 = axs[0].contourf(gridx, grid_y, (pred>0.5).reshape(100, 100), cmap="BrBG", alpha=0.3)
fig.colorbar(cax1, ax=axs[0], ticks=[0, 0.1, 0.5, 0.9, 1])



padding = 1.5
x_bounds = (map_model[0].weight.data[0,0] - padding, map_model[0].weight.data[0,0] + padding)
y_bounds = (map_model[0].weight.data[0,1] - padding, map_model[0].weight.data[0,1] + padding)


axs[1].plot(map_tracking[:,0], map_tracking[:,1], label="MAP")
axs[2].plot(map_tracking[:,0], map_tracking[:,1], label="MAP")
axs[1].set_title("Negative Log-Lik")

n_weight_samples = 200
weight_space_x = torch.linspace(x_bounds[0], x_bounds[1], n_weight_samples)
weight_space_y = torch.linspace(y_bounds[0], y_bounds[1], n_weight_samples)
weight_test_model = nn.Sequential(nn.Linear(2, 1, bias=False))
# print("shape", weight_test_model[0].weight.data.shape)
log_posterior_grid = torch.zeros((n_weight_samples, n_weight_samples))
log_lik_grid = torch.zeros((n_weight_samples, n_weight_samples))
with torch.no_grad():
    for id1, weight1 in enumerate(weight_space_x):
        for id2, weight2 in enumerate(weight_space_y):
            new_weights = torch.tensor([[weight1, weight2]])
            # print(new_weights.shape)
            weight_test_model[0].weight.data = new_weights
            log_lik = - loss_fun(torch.sigmoid(weight_test_model(train_data)), train_target)
            log_prior = weight_prior.log_prob(new_weights)
            log_posterior = log_lik + log_prior
            log_lik_grid[id1, id2] = log_lik
            log_posterior_grid[id1, id2] = log_posterior

cax2 = axs[1].contourf(weight_space_x, weight_space_y, log_lik_grid, cmap="BrBG", alpha=0.3, levels=10)
# fig.colorbar(cax2, ax=axs[1], ticks=torch.linspace(log_lik_grid.min(), log_lik_grid.max(), 5))
cax3 = axs[2].contourf(weight_space_x, weight_space_y, log_posterior_grid, cmap="BrBG", alpha=0.3, levels=10)
fig.colorbar(cax3, ax=axs[2], ticks=torch.linspace(log_posterior_grid.min(), log_posterior_grid.max(), 5))
axs[2].set_title("Unnormalized negative log posterior")



# laplace model
# estimate the hessian
# shapes: dim x n_datapoints @ n_datapoints x n_datapoints @ n_datapoints x dim, see Murphy chapter 8.3.1, page 247
predictions = torch.sigmoid(map_model(train_data)).squeeze()
log_lik_hessian = train_data.transpose(0,1) @ torch.diag(predictions * (1 - predictions)) @ train_data # hessian of negative log likelihood
log_prior_hessian = 2 / weight_prior_variance**2 # hessian of negative log gaussian
laplace_covariance = torch.inverse(log_lik_hessian + log_prior_hessian)
print(laplace_covariance)
laplace_distribution = torch.distributions.MultivariateNormal(map_model[0].weight.squeeze(), laplace_covariance)

laplace_std_visual = Ellipse(map_model[0].weight.data.squeeze(), 2*laplace_covariance[0,0], 2*laplace_covariance[1,1], color="blue", alpha=0.5, facecolor='none')
axs[2].add_patch(laplace_std_visual)


laplace_log_posterior_grid = torch.zeros((n_weight_samples, n_weight_samples))
with torch.no_grad():
    for id1, weight1 in enumerate(weight_space_x):
        for id2, weight2 in enumerate(weight_space_y):
            laplace_log_posterior_grid[id1, id2] = laplace_distribution.log_prob(torch.tensor([weight1, weight2]))
cax4 = axs[3].contourf(weight_space_x, weight_space_y, laplace_log_posterior_grid, cmap="BrBG", alpha=0.3, levels=10)
axs[3].set_title("Laplace approximation of posterior")


# define model
lvi_model = nn.Sequential(nn.Linear(2,1, bias=False))
lvi_model[0].weight.data = map_tracking[0].unsqueeze(dim=0) # copy weights
lvi_tracking = lvi_model[0].weight.detach().clone()
lvi_optimizer = torch.optim.SGD(lvi_model.parameters(), lr=1e-3)

weight_prior_mean = 0
weight_prior_variance = 2
weight_prior = torch.distributions.MultivariateNormal(torch.full((1,2), weight_prior_mean), weight_prior_variance * torch.eye(2))

for i in range(epochs):
    lvi_optimizer.zero_grad()
    pred = torch.sigmoid(lvi_model(train_data))
    # print(pred, train_target)
    log_lik = loss_fun(pred, train_target)
    log_prior = - weight_prior.log_prob(lvi_model[0].weight)
    log_joint = (log_prior + log_lik).mean()
    # print(log_joint)
    loss = log_joint + asdfghjkl.hessian(log_joint, lvi_model[0].weight, create_graph=True).log().sum()/n_datapoints
    loss.backward()
    lvi_optimizer.step()
    lvi_tracking = torch.cat([lvi_tracking, lvi_model[0].weight.detach().clone()], dim=0)
print(log_lik)

# print(lvi_tracking)
axs[1].plot(lvi_tracking[:,0], lvi_tracking[:,1], label="Laplace_VI")
axs[2].plot(lvi_tracking[:,0], lvi_tracking[:,1], label="Laplace_VI")
axs[1].legend()

plt.tight_layout()
plt.show()
