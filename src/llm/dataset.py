"""Dataset utilities for next-token training on JSONL or plain-text corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

DEFAULT_TOKENIZER_PATH = Path("data/tokenizer/tokenizer.json")
DEFAULT_CORPUS_PATH = Path("data/chunks.jsonl")


def _load_texts_from_jsonl(corpus_path: Path) -> list[str]:
    """Load valid `text` fields from JSONL corpus rows."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if corpus_path.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")

    texts: list[str] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError(f"No valid text rows found in JSONL corpus: {corpus_path}")
    return texts


def _load_texts_from_txt(corpus_path: Path) -> list[str]:
    """Load full plain-text corpus as a single text segment."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if corpus_path.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")

    text = corpus_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"No text content found in corpus: {corpus_path}")
    return [text]


def load_corpus_texts(corpus_path: Path) -> tuple[list[str], str]:
    """Load corpus text segments and return `(texts, corpus_format)`."""
    suffix = corpus_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_texts_from_jsonl(corpus_path), "jsonl"
    if suffix == ".txt":
        return _load_texts_from_txt(corpus_path), "txt"
    raise ValueError(
        f"Unsupported corpus extension: {suffix}. "
        "Supported formats: .jsonl (chunk rows) and .txt (plain text)."
    )


def load_chunk_texts(corpus_path: Path) -> list[str]:
    """Backward-compatible wrapper for JSONL corpus loading."""
    texts, _ = load_corpus_texts(corpus_path)
    return texts


def encode_corpus(
    *,
    tokenizer: Tokenizer,
    texts: list[str],
    eos_token: str = "[EOS]",
    max_tokens: int | None = None,
) -> list[int]:
    """Encode chunk texts into one flattened token-id stream."""
    eos_id = tokenizer.token_to_id(eos_token)
    token_ids: list[int] = []

    for text in texts:
        encoded_ids = tokenizer.encode(text).ids
        token_ids.extend(encoded_ids)
        if eos_id is not None:
            token_ids.append(int(eos_id))
        if max_tokens is not None and len(token_ids) >= max_tokens:
            break

    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    if not token_ids:
        raise ValueError("Encoded token stream is empty.")
    return token_ids


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window dataset for autoregressive next-token prediction."""

    def __init__(
        self,
        *,
        tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
        corpus_path: Path = DEFAULT_CORPUS_PATH,
        block_size: int = 384,
        stride: int = 128,
        max_tokens: int | None = None,
    ) -> None:
        """Create training pairs `(x, y)` with shifted token windows."""
        if block_size <= 0:
            raise ValueError("block_size must be > 0.")
        if stride <= 0:
            raise ValueError("stride must be > 0.")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        self.block_size = block_size
        self.stride = stride
        self.tokenizer_path = tokenizer_path
        self.corpus_path = corpus_path
        self.max_tokens = max_tokens
        self.corpus_format = ""

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        texts, self.corpus_format = load_corpus_texts(corpus_path)
        self.token_ids = encode_corpus(
            tokenizer=self.tokenizer,
            texts=texts,
            max_tokens=max_tokens,
        )
        if len(self.token_ids) <= self.block_size:
            raise ValueError(
                f"Token stream too short ({len(self.token_ids)} tokens) for block_size={self.block_size}. "
                "Lower block_size or provide more corpus text."
            )

        max_start = len(self.token_ids) - self.block_size - 1
        self.start_positions = list(range(0, max_start + 1, self.stride))
        if not self.start_positions:
            raise ValueError("No training windows were created. Adjust block_size/stride.")

    def __len__(self) -> int:
        """Return number of sliding windows."""
        return len(self.start_positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one training example `(x, y)` as int64 tensors."""
        start = self.start_positions[index]
        end = start + self.block_size
        x_ids = self.token_ids[start:end]
        y_ids = self.token_ids[start + 1 : end + 1]

        x = torch.tensor(x_ids, dtype=torch.long)
        y = torch.tensor(y_ids, dtype=torch.long)
        return x, y

    @property
    def vocab_size(self) -> int:
        """Tokenizer vocabulary size."""
        return int(self.tokenizer.get_vocab_size())

    def info(self) -> dict[str, Any]:
        """Return compact dataset stats for logging/debugging."""
        return {
            "tokenizer_path": str(self.tokenizer_path),
            "corpus_path": str(self.corpus_path),
            "corpus_format": self.corpus_format,
            "block_size": self.block_size,
            "stride": self.stride,
            "max_tokens": self.max_tokens,
            "vocab_size": self.vocab_size,
            "total_tokens": len(self.token_ids),
            "num_examples": len(self),
        }
