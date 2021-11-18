
import dataclasses
from lib2to3.pytree import Base
from mimetypes import init
from typing import Any
import torch
from torch import nn
from torch._C import dtype
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from enum import Enum
from datetime import datetime, time
import os
import math
import json
from pydantic.dataclasses import dataclass
from pydantic import Field, BaseModel, validator
from abc import ABC

class Log_likelihood_type(str, Enum):
    CATEGORICAL = "categorical"
    CATEGORICAL_2_CLASSES = "categorical_2_classes"
    MSE = "mse"
    MSE_NO_NOISE = "mse_no_noise"



@dataclass(init=True)
class LLVINetwork(nn.Module):
    """Base class for Last-Layer Variational Inference Networks (LLVI).
    """
    bias = False
    feature_dim: int
    out_dim: int
    feature_extractor: Any
    
    prior_mu: Any
    prior_log_var: Any

    optimizer_type: Any = torch.optim.SGD
    tau = 1
    lr = 1e-2

    @validator("prior_mu", "prior_log_var")
    def create_tensor_parameter(cls, x, values):
        feature_dim = values["feature_dim"]
        out_dim = values["out_dim"]
        return LLVINetwork.create_full_tensor(feature_dim, out_dim, x)

    def create_full_tensor(dim1, dim2, fill_value):
        return nn.Parameter(torch.full((dim1, dim2), fill_value=fill_value, requires_grad=True, dtype=torch.float32))

    def create_optimizer(optimizer_type, parameters, lr, **kwargs):
        return optimizer_type(parameters, lr=lr, **kwargs)


    def __post_init_post_parse__(self):
        self.prior_optimizer = LLVINetwork.create_optimizer(self.optimizer_type, [self.prior_mu, self.prior_log_var], self.lr)



        







    # def __init__(self, feature_extractor, feature_dim, out_dim, prior_mu=0, prior_log_var=1, lr=1e-2, tau=1, wdecay=0, bias=True, data_log_var=-1) -> None:
    #     super(LLVINetwork, self).__init__()
    #     self.bias = bias
    #     if self.bias:
    #         feature_dim += 1

    #     self.feature_dim = feature_dim
    #     self.out_dim = out_dim
    #     self.feature_extractor: nn.Module = feature_extractor

        

    #     self.tau = tau
    #     self.feature_extractor_optimizer = optim.SGD(self.feature_extractor.parameters(), lr=lr, momentum=0.8, weight_decay=wdecay)

    #     self.prior_mu = nn.Parameter(torch.full((feature_dim, out_dim), fill_value=prior_mu, requires_grad=True, dtype=torch.float32))
    #     self.prior_log_var = nn.Parameter(torch.full((feature_dim, out_dim), fill_value=prior_log_var, requires_grad=True, dtype=torch.float32))
    #     hyperparams = [self.prior_mu, self.prior_log_var]
    #     if data_log_var and (loss == Log_likelihood_type.MSE):
    #         self.data_log_var = nn.Parameter(torch.tensor([data_log_var], dtype=torch.float32), requires_grad=True) # the log variance of the data
    #         hyperparams += [self.data_log_var]
    #     self.prior_optimizer = optim.SGD(hyperparams, lr=lr, momentum=0.8) # optimizer for prior

    #     self.model_config = {"feature_dim": feature_dim, "out_dim": out_dim, "prior_mu": prior_mu, "prior_log_var": prior_log_var, "lr": lr,
    #     "tau": tau, "wdecay": wdecay, "bias": bias, "loss": loss.value, "data_log_var": data_log_var}
