import torch
from backpack import backpack, extend
from backpack.extensions import DiagHessian, DiagGGNExact, BatchGrad, DiagGGNMC
from torch.autograd import grad
from torch import nn

torch.manual_seed(1)
x = torch.tensor([[1,2],
                 [3, 4]], dtype=torch.float32)
y = torch.tensor([3, 7], dtype=torch.float32)

# (x_1 w_1 - y_1)^2 + 
# first order = 2(x w - y) x = 2x^2 w - xy
# second 2x^2

# second order nabla^2_f p(y|f) = nabla^2_f (y - f)^2 = 2y
# first order nabla_theta f(x, theta) = x

# x_1 0     1 0    x_1 0 
# 0 x_2     0 1    0 x_2


model = nn.Sequential(nn.Linear(2, 1)) 
print(model)
loss_fun = torch.nn.MSELoss()#nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters())
extend(model)
extend(loss_fun)
with backpack(DiagGGNExact(), BatchGrad(), DiagGGNMC(), DiagHessian()):
    pred = model(x)
    loss = loss_fun(pred, y)
    loss.backward()
print(model[0].weight)
print("Hessian", model[0].weight.diag_h)
print("GGN", model[0].weight.diag_ggn_exact)
print("GGN MC", model[0].weight.diag_ggn_mc)
print("Grad", model[0].weight.grad)
print("BATCH grad", model[0].weight.grad_batch)
print("Batch_grad square", 0.5 * model[0].weight.grad_batch.square().sum(dim=0))




my_hess = x.square().sum(dim=0)
print("My hessian", my_hess)
my_grad = ((model(x) - y) * x)
print("My gradient", my_grad)

# for i in range(1000):
#     optimizer.zero_grad()
#     pred = model(x)
#     loss = loss_fun(pred, y)
#     loss.backward()
#     optimizer.step()
#     print(loss.item())

# print(model.linear.weight)