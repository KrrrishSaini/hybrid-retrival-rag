"""Train a BPE tokenizer on `data/chunks.jsonl` corpus text."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers

LOGGER = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = Path("data/chunks.jsonl")
DEFAULT_OUT_DIR = Path("data/tokenizer")


def iter_chunk_texts(corpus_path: Path) -> Iterator[str]:
    """Yield chunk text strings from a JSONL corpus file."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if corpus_path.stat().st_size == 0:
        raise ValueError(f"Corpus file is empty: {corpus_path}")

    with corpus_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                item = json.loads(payload)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at line %d: %s", line_number, exc)
                continue
            text = str(item.get("text", "")).strip() if isinstance(item, dict) else ""
            if text:
                yield text


def train_bpe_tokenizer(
    corpus_path: Path,
    out_dir: Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Path:
    """Train and save a BPE tokenizer from corpus text."""
    texts = list(iter_chunk_texts(corpus_path))
    if not texts:
        raise ValueError(f"No usable text found in corpus: {corpus_path}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    sample_text = texts[0]
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)

    print(f"Tokenizer saved to: {tokenizer_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"Sample encoded sequence (first 40 ids): {encoded.ids[:40]}")
    print(f"Decoded reconstruction (first 300 chars): {decoded[:300]}")
    return tokenizer_path


def parse_args() -> argparse.Namespace:
    """Parse tokenizer training CLI arguments."""
    parser = argparse.ArgumentParser(description="Train BPE tokenizer from chunks.jsonl.")
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to chunks JSONL corpus.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for tokenizer artifact.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=8000,
        help="BPE vocabulary size.",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum token pair frequency to merge.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    try:
        train_bpe_tokenizer(
            corpus_path=args.corpus_path,
            out_dir=args.out_dir,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
