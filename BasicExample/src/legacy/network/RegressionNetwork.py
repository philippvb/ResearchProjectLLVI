import torch
from network import LLVINetwork
from pydantic.dataclasses import dataclass
from dataclasses import field
from typing import Any

@dataclass
class RegressionNetworkNoise(LLVINetwork):
    data_log_var: Any

    def __post_init_post_parse__(self):
        self.prior_optimizer = LLVINetwork.create_optimizer(self.optimizer_type, [self.prior_mu, self.prior_log_var, self.data_log_var], self.lr)

    
        
