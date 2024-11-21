import torch
import tiktoken
import torch.nn.functional as F
from datasets import load_dataset

from gpt import ModelConfig, GenerationConfig, Model


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = Model(ModelConfig()).to(device)
optimizer = model.configure_optimizer(lr=3e-4)
tokenizer = tiktoken.get_encoding("r50k_base")
