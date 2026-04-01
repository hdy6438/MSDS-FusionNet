import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class MoEMambaBlock(nn.Module):
    def __init__(self, d_model):
        super(MoEMambaBlock, self).__init__()
        self.mamba = Mamba(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)


    def forward(self, x):
        x = x + self.mamba(x)
        x = self.norm(x)
        x = F.leaky_relu(x)
        return x
