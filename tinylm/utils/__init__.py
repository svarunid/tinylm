import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReLU, LeakyReLU, PReLU, SiLU, GELU, GLU, LayerNorm, RMSNorm
from torch.optim import Adam, AdamW, RMSprop


class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()

    def forward(self, x, gate):
        return F.silu(x) * gate
