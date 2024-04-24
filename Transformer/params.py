from dataclasses import dataclass


@dataclass
class ModelArgs:
    dim: int = 512
    n_layers: int = 4
    n_heads: int = 8
    # n_kv_heads: int = 32 GQA
    vocab_size: int = 1000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 512
    dropout: float = 0.0
    hidden_dim: int = 256
