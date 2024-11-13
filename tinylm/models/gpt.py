import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_activation(config):
    params = config["activation"].get("parameters", {})

    match config["activation"]["name"].lower():
        case "gelu":
            return nn.GELU(**params)
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU(**params)
        case "prelu":
            return nn.PReLU(**params)
        case "silu":
            return nn.SiLU()
        case "glu":
            return nn.GLU(**params)


class PositionalEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        name = config["positional"]["name"]
        seq, emb = config["seq"], config["emb"]
        theta = config["positional"]["parameters"].get("theta", 10000)

        position = torch.arange(0, seq).unsqueeze(-1).expand(-1, emb)
        position = position / torch.pow(theta, 2 * (torch.arange(0, emb) // 2) / emb)
        if name == "absolute":
            position[:, 0::2] = torch.sin(position[:, 0::2])
            position[:, 1::2] = torch.cos(position[:, 1::2])
            self.register_buffer("embedding", position)
        elif name == "rotatory":
            sin, cos = torch.sin(position[:, 0::2]), torch.cos(position[:, 1::2])
            sin = torch.stack([sin, sin], dim=-1).reshape(seq, emb)
            cos = torch.stack([cos, cos], dim=-1).reshape(seq, emb)
            self.register_buffer("embedding", torch.cat([sin, cos], dim=-1))
        else:
            raise ValueError(f"Unknown positional encoding type: {name}")

    def forward(self, x):
        if self.config["positional"]["name"] == "absolute":
            return x + self.embedding
        else:
            sentinel = self.config["emb"] // 2 + self.config["emb"] % 2
            sin, cos = self.embedding[:, :sentinel], self.embedding[:, sentinel:]
            x = x * cos + torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x) * sin


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["emb"] % config["heads"] == 0
        self.config = config
        self.num_heads = config["heads"]
        self.head_dim = config["emb"] // config["heads"]

        self.query = nn.Linear(config["emb"], config["emb"])

        self.enable_gqa = "gqa" in config
        if self.enable_gqa:
            self.kv_heads = config["gqa"]["kv_heads"]
            self.key = nn.Linear(config["emb"], self.kv_heads * self.head_dim)
            self.value = nn.Linear(config["emb"], self.kv_heads * self.head_dim)
        else:
            self.key = nn.Linear(config["emb"], config["emb"])
            self.value = nn.Linear(config["emb"], config["emb"])

        self.proj = nn.Linear(config["emb"], config["emb"])

    def forward(self, x):
        B, T, C = x.size()

        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.kv_heads if self.enable_gqa else self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.kv_heads if self.enable_gqa else self.head_dim).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.head_dim**0.5, enable_gqa=self.enable_gqa)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["emb"])
        self.attn = MultiHeadAttention(config)
        self.fc1 = nn.Linear(config["emb"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["emb"])
        self.act = _get_activation(config)
        self.ln_2 = nn.LayerNorm(config["emb"])

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.fc2(self.act(self.fc1(x)))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config["vocab_size"], config["emb"])
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["layers"])])
        self.ln_f = nn.LayerNorm(config["emb"])
        self.output_proj = nn.Linear(config["emb"], config["vocab_size"], bias=False)

        if config["positional"]["name"] == "absolute":
            self.pos_embedding = PositionalEmbedding(config)

        if config["tie_weights"]:
            self.output_proj.weight = self.token_embedding.weight

    def forward(self, x):
        B, T = x.size()
        x = self.embedding(x)

        if self.config["positional"]["name"] == "absolute":
            x = self.pos_embedding(x)

        for block in self.blocks:
            x = block(x)

        return self.output_proj(self.ln_f(x))
