"""Autoregressive text generation with the custom decoder-only Transformer."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .model import DecoderOnlyTransformer, TransformerConfig

LOGGER = logging.getLogger(__name__)


def smart_decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    """Decode token IDs with proper subword joining.

    The BPE tokenizer with a Whitespace pre-tokenizer naively inserts
    spaces between all subtokens.  This function uses the tokenizer's
    encode-with-offsets on a joined token string to recover which tokens
    belong to the same word.
    """
    if not ids:
        return ""

    # Get raw token strings (skip specials)
    raw_tokens: list[str] = []
    for tid in ids:
        t = tokenizer.id_to_token(tid)
        if t and not t.startswith("["):
            raw_tokens.append(t)
    if not raw_tokens:
        return ""

    # Use naive decode — the post-processing in clean_raw_output handles
    # BPE fragmentation.  This avoids complex heuristics.
    return tokenizer.decode(ids, skip_special_tokens=True)

DEFAULT_MODEL_PATH = Path("data/llm/model.pt")
DEFAULT_CONFIG_PATH = Path("data/llm/config.json")


def auto_select_device(requested_device: str = "auto") -> torch.device:
    """Select generation device with preference mps -> cuda -> cpu."""
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model_and_tokenizer(
    *,
    model_path: Path = DEFAULT_MODEL_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
    tokenizer_path: Path | None = None,
    device: str = "auto",
) -> tuple[DecoderOnlyTransformer, Tokenizer, torch.device]:
    """Load trained model checkpoint, config, and tokenizer."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    model_cfg_payload = payload.get("model")
    if not isinstance(model_cfg_payload, dict):
        raise ValueError(f"Invalid model config in {config_path}")

    resolved_tokenizer_path = tokenizer_path or Path(str(payload.get("tokenizer_path", "")))
    if not resolved_tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {resolved_tokenizer_path}. "
            "Pass --tokenizer_path or retrain tokenizer."
        )

    model_cfg = TransformerConfig(**model_cfg_payload)
    runtime_device = auto_select_device(device)
    model = DecoderOnlyTransformer(model_cfg)

    checkpoint: Any = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if state_dict is None:
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unsupported checkpoint format: {model_path}")
    model.load_state_dict(state_dict)
    model.to(runtime_device)
    model.eval()

    tokenizer = Tokenizer.from_file(str(resolved_tokenizer_path))
    return model, tokenizer, runtime_device


@torch.no_grad()
def generate_text(
    *,
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.8,
    top_k: int = 50,
    stop_strings: list[str] | None = None,
    device: torch.device,
) -> str:
    """Generate text from prompt using causal autoregressive decoding."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0.")

    encoded_prompt = tokenizer.encode(prompt).ids
    if not encoded_prompt:
        unk_id = tokenizer.token_to_id("[UNK]")
        if unk_id is None:
            raise ValueError("Prompt encodes to empty sequence and tokenizer has no [UNK] token.")
        encoded_prompt = [int(unk_id)]

    tokens = torch.tensor([encoded_prompt], dtype=torch.long, device=device)
    prompt_len = len(encoded_prompt)
    stop_markers = [marker for marker in (stop_strings or []) if marker]
    stopped_text: str | None = None

    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.block_size :]
        logits, _ = model(context)
        next_token_logits = logits[:, -1, :]

        if temperature <= 0:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = next_token_logits / temperature
            if top_k > 0:
                k = min(top_k, scaled_logits.shape[-1])
                values, _ = torch.topk(scaled_logits, k=k, dim=-1)
                threshold = values[:, -1].unsqueeze(-1)
                scaled_logits = torch.where(
                    scaled_logits < threshold,
                    torch.full_like(scaled_logits, float("-inf")),
                    scaled_logits,
                )
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)
        if stop_markers:
            generated_ids = tokens[0].tolist()[prompt_len:]
            decoded_generated = tokenizer.decode(generated_ids, skip_special_tokens=True)
            # Normalize for stop-string matching: collapse whitespace, check
            # both original and collapsed forms (BPE may insert spaces, e.g.
            # "<END>" decoded as "< end >").
            collapsed = re.sub(r"\s+", " ", decoded_generated)
            collapsed_lower = collapsed.lower()
            stop_positions: list[int] = []
            for marker in stop_markers:
                marker_collapsed = re.sub(r"\s+", "", marker).lower()
                # Try exact match first, then collapsed match
                pos = decoded_generated.find(marker)
                if pos >= 0:
                    stop_positions.append(pos)
                    continue
                # Search for the marker with spaces removed in the collapsed text
                coll_pos = collapsed_lower.find(marker_collapsed)
                if coll_pos >= 0:
                    # Map collapsed position back to original position approximately
                    stop_positions.append(max(0, coll_pos - 1))
            if stop_positions:
                cut_at = min(stop_positions)
                stopped_text = decoded_generated[:cut_at].rstrip()
                break

    if stopped_text is not None:
        return stopped_text

    generated_ids = tokens[0].tolist()[prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    """Parse generation CLI args."""
    parser = argparse.ArgumentParser(description="Generate text with custom mini Transformer.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model config JSON.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=None,
        help="Optional tokenizer path override.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Max generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling cutoff (0 disables).")
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=None,
        help='Optional stop strings, e.g. --stop_strings "<END_EXAMPLE>" "CONTEXT:" "QUESTION:"',
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cpu|mps|cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for generation."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(args.seed)

    try:
        model, tokenizer, device = load_model_and_tokenizer(
            model_path=args.model_path,
            config_path=args.config_path,
            tokenizer_path=args.tokenizer_path,
            device=args.device,
        )
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_strings=args.stop_strings,
            device=device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Prompt:")
    print(args.prompt)
    print("-" * 100)
    print("Generated:")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
