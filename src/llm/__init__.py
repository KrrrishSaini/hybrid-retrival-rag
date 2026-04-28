"""Custom decoder-only LLM package for local RAG generation."""

from .dataset import NextTokenDataset
from .model import DecoderOnlyTransformer, TransformerConfig

__all__ = [
    "DecoderOnlyTransformer",
    "NextTokenDataset",
    "TransformerConfig",
]
