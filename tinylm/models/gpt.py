import torch
import torch.nn as nn
import torch.nn.functional as F

from tinylm import utils


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, seq, theta, name="absolute"):
        super().__init__()
        self.dim, self.name = dim, name
        # Create a tensor of shape (seq, dim) with positional indices
        position = torch.arange(0, seq).unsqueeze(-1).expand(-1, dim)
        # Compute the positional encodings
        position = position / torch.pow(theta, 2 * (torch.arange(0, dim) // 2) / dim)
        if name == "absolute":
            position[:, 0::2] = torch.sin(position[:, 0::2])
            position[:, 1::2] = torch.cos(position[:, 1::2])
            self.register_buffer("emb", position)
        elif name == "rotary":
            # Compute sine and cosine components for rotary embeddings
            sin, cos = torch.sin(position[:, 0::2]), torch.cos(position[:, 1::2])
            # Duplicate and reshape to match the required dimensions
            sin = torch.stack([sin, sin], dim=-1).reshape(seq, dim)
            cos = torch.stack([cos, cos], dim=-1).reshape(seq, dim)
            self.register_buffer("sin", sin)
            self.register_buffer("cos", cos)
        else:
            raise ValueError(f"Unknown positional encoding type: {name}")

    def forward(self, x):
        if self.name == "absolute":
            return x + self.emb
        else:
            return x * self.cos + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x) * self.sin


class CausalAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads=None, bias=False):
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

        self.q_proj = nn.Linear(dim, dim, bias)
        self.k_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias)
        self.v_proj = nn.Linear(dim, self.kv_heads * self.head_dim, bias)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None, pos_emb=None):
        B, T, C = x.size()

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        if pos_emb:
            # If provide, apply rotary positional embeddings
            q, k = pos_emb(q), pos_emb(k)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = mask * causal_mask if mask is not None else causal_mask

        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool(), enable_gqa=self.enable_gqa)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.o_proj(attn_output)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["dim"]
        norm, activation = config["norm"], config["activation"]
        kv_heads = config["gqa"]["kv_heads"] if "gqa" in config else None

        # Dynamically initialize the input normalization layer based on configuration
        self.input_norm = getattr(utils, norm["name"])(dim, **norm.get("parameters", {}))
        self.attn = CausalAttention(dim, config["heads"], kv_heads, config["qkv_bias"])
        self.attn_norm = getattr(utils, norm["name"])(dim, **norm.get("parameters", {}))

        self.up_proj = nn.Linear(dim, config["hidden_size"])
        # Dynamically initialize the activation function
        self.act = getattr(utils, activation["name"])(**activation.get("parameters", {}))
        self.down_proj = nn.Linear(config["hidden_size"], dim)

    def forward(self, x, mask=None, pos_emb=None):
        x = x + self.attn(self.input_norm(x), mask, pos_emb)
        x = x + self.down_proj(self.act(self.up_proj(self.attn_norm(x))))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim, seq, vocab, positional = config["dim"], config["seq"], config["vocab_size"], config["positional"]

        self.tok_emb = nn.Embedding(vocab, dim)
        # Positional embedding layer (absolute or rotary based on configuration)
        self.pos_emb = PositionalEmbedding(dim, seq, positional["parameters"]["theta"], positional["name"])

        self.blocks = nn.ModuleList([Block(config) for _ in range(config["layers"])])
        self.norm = getattr(utils, config["norm"]["name"])(dim, **config["norm"].get("parameters", {}))
        self.output_proj = nn.Linear(dim, vocab, bias=False)

        if config["tie_weights"]:
            self.output_proj.weight = self.tok_emb.weight

    def forward(self, x, mask=None):
        B, T = x.size()
        x = self.tok_emb(x)

        pos_emb = None
        if self.config["positional"]["name"] == "absolute":
            x = self.pos_emb(x)
        else:
            # For rotary embeddings, pass the positional embedding layer to the blocks
            pos_emb = self.pos_emb

        for block in self.blocks:
            x = block(x, mask=mask, pos_emb=pos_emb)

        return self.output_proj(self.norm(x))
