from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def precompute_rope_embeddings(dim, seq, theta=10000):
    position = torch.arange(0, seq).unsqueeze(-1).expand(-1, dim)
    div_term = torch.pow(theta, -2 * (torch.arange(0, dim) // 2) / dim)
    position = position * div_term

    sin = torch.sin(position)
    cos = torch.cos(position)
    return sin, cos


def apply_rope(q, k, sin, cos):
    q = q * cos + torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape_as(q) * sin
    k = k * cos + torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape_as(k) * sin
    return q, k


@dataclass
class GPTConfig:
    dim: int
    seq: int
    heads: int
    layers: int
    hidden_size: int
    vocab_size: int
    theta: int = 100000
    qkv_bias: bool = False


class CausalAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None, bias=False):
        assert dim % heads == 0
        assert heads % kv_heads == 0

        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.kv_heads = kv_heads

        self.q_proj = nn.Linear(dim, dim, bias)
        self.k_proj = nn.Linear(dim, int(self.kv_heads * self.head_dim), bias)
        self.v_proj = nn.Linear(dim, int(self.kv_heads * self.head_dim), bias)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, sin, cos, mask=None):
        B, T, C = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        q, k = apply_rope(q, k, sin, cos)
        q, k, v = map(lambda x: x.view(B, T, -1, self.head_dim).transpose(1, 2), (q, k, v))

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = mask * causal_mask if mask is not None else causal_mask

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool(), enable_gqa=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(attn_output)


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.input_norm = nn.LayerNorm(dim)
        self.up_proj = nn.Linear(dim, hidden)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.down_proj(self.up_proj(self.input_norm(x)) * F.silu(self.gate_proj(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.dim)
        self.attn = CausalAttention(config.dim, config.heads, config.kv_heads, config.qkv_bias)
        self.mlp = MLP(config.dim, config.hidden)

    def forward(self, x, sin, cos, mask=None):
        x = x + self.attn(self.input_norm(x), sin, cos, mask)
        return x + self.mlp(x)


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.emb = nn.Embedding(config.vocab, config.dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(config.dim)
        self.output_proj = nn.Linear(config.dim, config.vocab, bias=False)

        # Tie the weights of the embedding and the last projection layer.
        self.output_proj.weight = self.emb.weight

        sin, cos = precompute_rope_embeddings(config.dim, config.seq, config.theta)
        self.register_buffer("sin", sin)
        self.register_buffer("cos", cos)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)

    def configure_optimizer(self, lr, eps=1e-8, weight_decay=1e-2):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}]
        return optim.AdamW(optim_groups, lr, eps=eps)

    def forward(self, x, mask=None):
        B, T = x.size()
        x = self.emb(x)

        for block in self.blocks:
            x = block(x, self.sin, self.cos, mask)

        return self.output_proj(self.norm(x))