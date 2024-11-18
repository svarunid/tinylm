# TinyLM

Building and training my own tiny langugae model from scratch. The model achitecture is blend between architectures
of popular open-source models like LLaMA, Qwen2.5-Coder, etc. The architecture uses RoPE for positional encoding,
Grouped Query Attention (GQA), QKV bias, weight tying, etc.

## Future Enhancements

- Initialize weights according to Maximal Update Parameterization (muP).
