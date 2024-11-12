import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding layers
        self.token_embedding = nn.Embedding(config["vocab_size"], config["emb"])

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config["layers"])])

        # Output layer
        self.ln_f = nn.LayerNorm(config["emb"])
        self.output_proj = nn.Linear(config["emb"], config["vocab_size"], bias=False)

        if config["tie_weights"]:
            self.output_proj.weight = self.token_embedding.weight

    def _get_positional_embedding(self, config):
        if config["positional"]["name"] == "Sinusoid":
            theta = config["positional"]["parameters"].get("theta", 10000)
            position = torch.arange(0, config["seq"]).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config["emb"], 2) * -(math.log(theta) / config["emb"]))
            pe = torch.zeros(1, config["seq"], config["emb"])
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_embedding", pe)
        elif config["positional"]["name"] == "RoPE":
            theta = config["positional"]["parameters"].get("theta", 10000)
            position = torch.arange(0, config["seq"]).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, config["emb"], 2) * -(math.log(theta) / config["emb"]))
            pe = torch.zeros(1, config["seq"], config["emb"])
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer("position_embedding", pe)
        elif config["positional"]["name"] == "Relative":
            # Placeholder for relative positional embeddings if needed later
            pe = torch.zeros(1, config["seq"], config["emb"])
            self.register_buffer("position_embedding", pe)
        else:
            raise ValueError(f"Unknown positional encoding type: {config['positional']['name']}")

    def forward(self, x):
        B, T = x.size()
        token_embeddings = self.token_embedding(x)  # (B, T, emb)
        position_embeddings = self.position_embedding[:, :T, :]  # (1, T, emb)
        x = token_embeddings + position_embeddings  # (B, T, emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.output_proj(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["emb"])
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config["emb"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config["emb"] % config["heads"] == 0
        self.num_heads = config["heads"]
        self.head_dim = config["emb"] // config["heads"]
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(config["emb"], config["emb"])
        self.key = nn.Linear(config["emb"], config["emb"])
        self.value = nn.Linear(config["emb"], config["emb"])
        self.proj = nn.Linear(config["emb"], config["emb"])

        self.enable_gqa = "gqa" in config
        if self.enable_gqa:
            self.kv_heads = config["gqa"]["kv_heads"]
            self.query = nn.Linear(config["emb"], config["emb"])
            self.key = nn.Linear(config["emb"], self.kv_heads * self.head_dim)
            self.value = nn.Linear(config["emb"], self.kv_heads * self.head_dim)

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, scale=self.scale, enable_gqa=self.enable_gqa)  # (B, heads, T, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        return self.proj(attn_output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["emb"], config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], config["emb"])
        self.activation = getattr(F, config["activation"]["name"].lower())
        self.activation_params = config["activation"].get("parameters", {})

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x, **self.activation_params)
        x = self.fc2(x)
        return x
