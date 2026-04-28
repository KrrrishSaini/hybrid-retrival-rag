"""Train a mini decoder-only Transformer on chunk corpus text."""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import DEFAULT_CORPUS_PATH, DEFAULT_TOKENIZER_PATH, NextTokenDataset
from .model import DecoderOnlyTransformer, TransformerConfig

LOGGER = logging.getLogger(__name__)
DEFAULT_OUT_DIR = Path("data/llm")


def auto_select_device(requested_device: str = "auto") -> torch.device:
    """Select device with preference: mps -> cuda -> cpu when `auto`."""
    if requested_device != "auto":
        return torch.device(requested_device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_perplexity(loss: float) -> float:
    """Compute perplexity safely for logging."""
    return math.exp(min(loss, 20.0))


def _append_training_log(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line to the training log."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train mini decoder-only Transformer from scratch.")
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to tokenizer.json",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Path to training corpus (.jsonl chunks or .txt QA corpus).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for model artifacts",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--block_size", type=int, default=384, help="Token block/window size.")
    parser.add_argument("--stride", type=int, default=128, help="Sliding window stride.")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Optional cap on total corpus tokens for quick runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto | cpu | mps | cuda",
    )
    parser.add_argument("--n_layers", type=int, default=4, help="Transformer decoder blocks.")
    parser.add_argument("--n_heads", type=int, default=4, help="Attention heads.")
    parser.add_argument("--n_embd", type=int, default=256, help="Embedding dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for training."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    torch.manual_seed(args.seed)
    device = auto_select_device(args.device)
    LOGGER.info("Training device selected: %s", device)

    try:
        dataset = NextTokenDataset(
            tokenizer_path=args.tokenizer_path,
            corpus_path=args.corpus_path,
            block_size=args.block_size,
            stride=args.stride,
            max_tokens=args.max_tokens,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    dataset_info = dataset.info()
    LOGGER.info("Corpus format detected: %s", dataset_info.get("corpus_format"))
    LOGGER.info("Dataset stats: %s", dataset_info)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    if len(dataloader) == 0:
        print("Error: DataLoader has zero batches. Check block_size/batch_size/corpus size.")
        return 1

    config = TransformerConfig(
        vocab_size=dataset.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = DecoderOnlyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "model.pt"
    config_path = args.out_dir / "config.json"
    training_log_path = args.out_dir / "training_log.jsonl"

    if training_log_path.exists():
        training_log_path.unlink()

    total_steps = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="step")

        for step, (x, y) in enumerate(progress, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, targets=y)
            if loss is None:
                raise RuntimeError("Training loss was None; targets were expected.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_steps += 1
            step_loss = float(loss.item())
            epoch_losses.append(step_loss)
            step_ppl = _safe_perplexity(step_loss)

            progress.set_postfix(loss=f"{step_loss:.4f}", ppl=f"{step_ppl:.2f}")
            LOGGER.info(
                "epoch=%d step=%d global_step=%d loss=%.6f perplexity=%.4f",
                epoch,
                step,
                total_steps,
                step_loss,
                step_ppl,
            )

            _append_training_log(
                training_log_path,
                {
                    "timestamp_unix": int(time.time()),
                    "epoch": epoch,
                    "step": step,
                    "global_step": total_steps,
                    "loss": step_loss,
                    "perplexity": step_ppl,
                },
            )

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        epoch_ppl = _safe_perplexity(epoch_loss)
        LOGGER.info(
            "epoch=%d done avg_loss=%.6f perplexity=%.4f steps=%d",
            epoch,
            epoch_loss,
            epoch_ppl,
            len(epoch_losses),
        )

    torch.save({"model_state_dict": model.state_dict()}, model_path)
    config_payload = {
        "model": config.to_dict(),
        "tokenizer_path": str(args.tokenizer_path),
        "corpus_path": str(args.corpus_path),
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "block_size": args.block_size,
            "stride": args.stride,
            "max_tokens": args.max_tokens,
            "device": str(device),
            "seed": args.seed,
            "total_steps": total_steps,
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    print(f"Saved model: {model_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved training log: {training_log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
