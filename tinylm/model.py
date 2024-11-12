from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int
    d_model: int
    n_layer: int
    n_heads: int

    def enable_gqa(self, kv_heads):
        self.kv_heads = kv_heads


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
