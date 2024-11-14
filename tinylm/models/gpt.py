from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_layer(config, layers):
    name = config["name"]
    params = config.get("parameters", {})

    if name in layers:
        return partial(getattr(nn, name), **params)
    else:
        raise ValueError(f"Unknown activation function: {name}. The name is case-sensitive.")


_get_norm = partial(_get_layer, layers=["LayerNorm", "RMSNorm"])
_get_activation = partial(_get_layer, layers=["ReLU", "LeakyReLU", "PReLU", "SiLU", "GLU", "GELU"])


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, seq, theta, name="absolute"):
        super().__init__()
        self.dim, self.name = dim, name
        position = torch.arange(0, seq).unsqueeze(-1).expand(-1, dim)
        position = position / torch.pow(theta, 2 * (torch.arange(0, dim) // 2) / dim)
        if name == "absolute":
            position[:, 0::2] = torch.sin(position[:, 0::2])
            position[:, 1::2] = torch.cos(position[:, 1::2])
            self.register_buffer("emb", position)
        elif name == "rotary":
            sin, cos = torch.sin(position[:, 0::2]), torch.cos(position[:, 1::2])
            sin = torch.stack([sin, sin], dim=-1).reshape(seq, dim)
            cos = torch.stack([cos, cos], dim=-1).reshape(seq, dim)
            self.register_buffer("emb", torch.cat([sin, cos], dim=-1))
        else:
            raise ValueError(f"Unknown positional encoding type: {name}")

    def forward(self, x):
        if self.name == "absolute":
            return x + self.emb
        else:
            sentinel = self.dim // 2 + self.dim % 2
            sin, cos = self.emb[:, :sentinel], self.emb[:, sentinel:]
            return x * cos + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.size()) * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None):
        super().__init__()
        assert dim % heads == 0
        self.num_heads = heads
        self.head_dim = dim // heads

        if kv_heads is not None:
            assert heads % kv_heads == 0
            self.enable_gqa = True
            self.kv_heads = kv_heads
        else:
            self.enable_gqa = False
            self.kv_heads = heads

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.value = nn.Linear(dim, self.kv_heads * self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None, pos_emb=None):
        B, T, C = x.size()

        q, k, v = self.query(x), self.key(x), self.value(x)
        if pos_emb:
            q, k = pos_emb(q), pos_emb(k)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=self.enable_gqa)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = _get_norm(config["norm"])(config["dim"])
        kv_heads = config["gqa"]["kv_heads"] if "gqa" in config else None
        self.attn = MultiHeadAttention(config["dim"], config["heads"], kv_heads)

        self.fc1 = nn.Linear(config["dim"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["dim"])
        self.act = _get_activation(config["activation"])()
        self.ln_2 = _get_norm(config["norm"])(config["dim"])

    def forward(self, x, mask=None, pos_emb=None):
        x = x + self.attn(self.ln_1(x), mask, pos_emb)
        x = x + self.fc2(self.act(self.fc1(self.ln_2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, seq, vocab, positional = config["dim"], config["seq"], config["vocab_size"], config["positional"]

        self.tok_embedding = nn.Embedding(vocab, dim)
        self.pos_embedding = PositionalEmbedding(dim, seq, positional.get("theta", 10000), positional["name"])

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["layers"])])
        self.ln_f = _get_norm(config["norm"])(dim)
        self.output_proj = nn.Linear(dim, vocab, bias=False)

        if config["tie_weights"]:
            self.output_proj.weight = self.tok_embedding.weight

    def forward(self, x, mask=None):
        B, T = x.size()
        x = self.tok_embedding(x)

        pos_emb = None
        if self.config["positional"]["name"] == "absolute":
            x = self.pos_embedding(x)
        else:
            pos_emb = self.pos_embedding

        for block in self.blocks:
            x = block(x, mask=mask, pos_emb=pos_emb)

        return self.output_proj(self.ln_f(x))
