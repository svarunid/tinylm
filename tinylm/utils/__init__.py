import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, LeakyReLU, PReLU, SiLU, GELU, GLU, LayerNorm, RMSNorm


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return F.silu(x) * self.gate_proj(x)
