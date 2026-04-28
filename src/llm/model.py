"""Decoder-only Transformer language model implemented from scratch in PyTorch."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TransformerConfig:
    """Configuration for the mini decoder-only Transformer."""

    vocab_size: int
    block_size: int = 384
    n_layers: int = 4
    n_heads: int = 4
    n_embd: int = 256
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Serialize config as a plain dict."""
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention for decoder-only modeling."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads.")

        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads

        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal attention. Input shape: (B, T, C)."""
        batch_size, seq_len, channels = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_scores = attn_scores.masked_fill(
            ~self.causal_mask[:, :, :seq_len, :seq_len],
            float("-inf"),
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        attn_output = self.resid_dropout(self.out_proj(attn_output))
        return attn_output


class FeedForward(nn.Module):
    """Position-wise feed-forward block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward projection."""
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single decoder block: LN -> self-attn -> residual, LN -> FFN -> residual."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one decoder block."""
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffn(self.ln_2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """Compact decoder-only Transformer LM."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize module parameters."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass. Returns `(logits, loss)`."""
        batch_size, seq_len = x.shape
        if seq_len > self.config.block_size:
            raise ValueError(
                f"Input length {seq_len} exceeds block_size={self.config.block_size}."
            )

        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.long)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions).unsqueeze(0)
        hidden = self.dropout(token_emb + pos_emb)

        for block in self.blocks:
            hidden = block(hidden)

        hidden = self.ln_f(hidden)
        logits = self.lm_head(hidden)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, self.config.vocab_size),
                targets.view(batch_size * seq_len),
            )
        return logits, loss
