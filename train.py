import tiktoken
from datasets import load_dataset

from gpt import GPTConfig, GPT

config = GPTConfig(
    dim=768,
    seq=1024,
    heads=12,
    kv_heads=8,
    layers=12,
    hidden_size=768 * 4,
    vocab_size=49152,
)

model = GPT(config).compile()

tokenizer = tiktoken.get_encoding("r50k_base")

optimizer = model.configure_optimizer(lr=3e-4)

ds = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", split="train", streaming=True)
