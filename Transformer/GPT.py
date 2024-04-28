import math
import struct
import inspect
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from params import ModelArgs
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor):
        var = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        hidden_states = hidden_states * torch.rsqrt(var)
        return self.weight * hidden_states


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])

    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    # tmp = xq.float().reshape(xq.shape[:-1] + (-1, 2))
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.flash_attention = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash_attention:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(self, x, freqs_cos, freqs_sin):
        bs, seqlen = x.shape[0], x.shape[1]
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        q = q.view(bs, seqlen, self.n_heads, self.head_dim)
        k = k.view(bs, seqlen, self.n_heads, self.head_dim, )
        v = v.view(bs, seqlen, self.n_heads, self.head_dim, )

        xq, xk = apply_rotary_emb(q, k, freqs_cos, freqs_sin)
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = v.transpose(1, 2)
        # flash implementation
        if self.flash_attention:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      is_causal=True)
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            # scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        # output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w2 = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.w3 = nn.Linear(args.hidden_dim, args.dim, bias=False)

    def forward(self, x):
        x = F.silu(self.w1(x)) * self.w2(x)
        x = self.w3(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.attention_rms = RMSNorm(args.dim, args.norm_eps)
        self.ffn = FeedForward(args)
        self.ffn_rms = RMSNorm(args.dim, args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        x = self.attention_rms(x)
        x = x + self.attention(x, freqs_cos, freqs_sin)
        x = self.ffn_rms(x)
        x = x + self.ffn(x)
        return x


class TransformerDecoderOnly(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.params = args
        self.rms = RMSNorm(args.dim)
        self.embedding = torch.nn.Embedding(args.vocab_size, args.dim)
        self.Decoder = torch.nn.ModuleList([TransformerBlock(i, args) for i in range(args.n_layers)])
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # share the embedding parameters with the embedding parameters
        self.embedding.weight = self.output.weight

        self.init_rope()

    def init_rope(self):
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, tokens):
        bsz, seqlen = tokens.shape
        freqs_cos = self.freqs_cos[: seqlen]
        freqs_sin = self.freqs_sin[: seqlen]
        x = self.embedding(tokens)
        for layer in self.Decoder:
            x = layer(x, freqs_cos, freqs_sin)
        x = self.rms(x)
        logits = self.output(x)
        return logits


if __name__ == '__main__':
    params = ModelArgs()
    model = TransformerDecoderOnly(params)
    print(model)
    data = torch.range(0, 9).unsqueeze(0).to(torch.long)
    result = model.forward(data)
    print(result.shape)
