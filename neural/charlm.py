"""Causal character transformer: wime's character network.

Pre-norm blocks, learned positions and an output head tied to the character
embedding. Parameter names are part of the checkpoint format: tok, pos,
blocks.N.{ln1,qkv,proj,ln2,fc,out}, ln_f; head.weight is tok.weight and is not
stored.
"""
import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F

# Token ids 0-2; id 0 also marks positions without a label.
RESERVED = ["<pad>", "<unk>", "<bos>"]


@dataclasses.dataclass
class Config:
    vocab: int = 8192
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    context: int = 64
    dropout: float = 0.1


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n = cfg.n_embd
        self.heads = cfg.n_head
        self.ln1, self.qkv, self.proj = nn.LayerNorm(n), nn.Linear(n, 3 * n), nn.Linear(n, n)
        self.ln2, self.fc, self.out = nn.LayerNorm(n), nn.Linear(n, 4 * n), nn.Linear(4 * n, n)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, n = x.shape
        q, k, v = (z.view(b, t, self.heads, -1).transpose(1, 2) for z in self.qkv(self.ln1(x)).split(n, -1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, n)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)))))


class CharLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.n_embd)
        self.pos = nn.Embedding(cfg.context, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab, bias=False)
        self.head.weight = self.tok.weight
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Residual projections shrink with depth so the stream stays bounded.
                residual = name.endswith((".proj", ".out"))
                nn.init.normal_(module.weight, std=0.02 / math.sqrt(2 * cfg.n_layer) if residual else 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """Logits [B, T, V] for IDX, and the mean loss against TARGETS when given."""
        x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device)))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=0)
