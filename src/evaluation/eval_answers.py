"""Answer-quality evaluation: token-F1, exact match, and faithfulness.

Runs the full RAG pipeline (retrieval + generation) on the QA dataset and
reports answer-level metrics alongside retrieval metrics.

Usage:
    python3 -m src.evaluation.eval_answers --use_reranker --top_k 3
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from src.llm.pretrained_generate import PretrainedGenerator
from src.rag.answer import ground_answer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, DEFAULT_RERANKER_MODEL
from src.retrieval.dense_baseline import DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME
from src.retrieval.bm25_baseline import DEFAULT_CHUNKS_PATH

LOGGER = logging.getLogger(__name__)
DEFAULT_QA_PATH = Path("data/qa_dataset.json")

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "to", "was", "were", "will", "with",
}


def _normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_tokens(text: str) -> list[str]:
    """Tokenize normalized text, removing stopwords."""
    return [tok for tok in _normalize_answer(text).split() if tok not in _STOPWORDS and len(tok) > 1]


def token_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 between prediction and reference."""
    pred_tokens = _get_tokens(prediction)
    ref_tokens = _get_tokens(reference)
    if not pred_tokens or not ref_tokens:
        return 1.0 if not pred_tokens and not ref_tokens else 0.0
    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 if normalized prediction matches reference, else 0.0."""
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(reference) else 0.0


def load_qa_dataset(qa_path: Path) -> list[dict[str, Any]]:
    """Load QA dataset entries that have gold_answer."""
    if not qa_path.exists():
        raise FileNotFoundError(f"QA dataset not found: {qa_path}")
    with qa_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("QA dataset must be a JSON list.")

    records: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        question = item.get("question", "")
        gold_answer = item.get("gold_answer", "")
        if not question or not gold_answer:
            continue
        records.append(item)
    return records


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality.")
    parser.add_argument("--qa_path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--chunks_path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index_dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--top_k", type=int, default=3, help="Chunks to retrieve per query.")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_context_chars", type=int, default=6000)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--sample_top_k", type=int, default=50)
    parser.add_argument("--llm_device", type=str, default="auto")
    parser.add_argument("--retrieval_device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only first N QA items (0=all).")
    parser.add_argument(
        "--log_level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(args.seed)

    qa_samples = load_qa_dataset(args.qa_path)
    if not qa_samples:
        print("No QA samples with gold_answer found. Add gold_answer to qa_dataset.json entries.")
        return 0

    if args.limit > 0:
        qa_samples = qa_samples[: args.limit]

    print(f"Evaluating {len(qa_samples)} QA samples with gold answers...")

    try:
        retriever = HybridRetriever(
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            alpha=args.alpha,
            device=args.retrieval_device,
            local_files_only=args.local_files_only,
        )
        reranker: CrossEncoderReranker | None = None
        if args.use_reranker:
            reranker = CrossEncoderReranker(
                device=args.retrieval_device,
                local_files_only=args.local_files_only,
            )

        generator = PretrainedGenerator(device=args.llm_device, max_new_tokens=args.max_new_tokens)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Error loading models: {exc}")
        return 1

    f1_scores: list[float] = []
    em_scores: list[float] = []
    faith_scores: list[float] = []

    for idx, sample in enumerate(qa_samples, start=1):
        question = sample["question"]
        gold_answer = sample["gold_answer"]

        # Retrieve
        if reranker:
            candidates = retriever.retrieve(query=question, top_k=20)
            chunks = reranker.rerank(query=question, candidates=candidates, top_k=args.top_k)
        else:
            chunks = retriever.retrieve(query=question, top_k=args.top_k)

        # Generate with pretrained Qwen2.5 model
        pipeline_answer = generator.generate_answer(
            question=question,
            chunks=chunks,
            max_new_tokens=args.max_new_tokens,
        )

        # Score the generated answer
        f1 = token_f1(pipeline_answer, gold_answer)
        em = exact_match(pipeline_answer, gold_answer)
        grounded = ground_answer(pipeline_answer, chunks)

        f1_scores.append(f1)
        em_scores.append(em)
        faith_scores.append(grounded["faithfulness"])

        print(
            f"  [{idx:>3}/{len(qa_samples)}] F1={f1:.3f} EM={em:.0f} "
            f"Faith={grounded['faithfulness']:.2f} | {question[:60]}...",
            flush=True,
        )

    print()
    print("=" * 60)
    print("ANSWER QUALITY SUMMARY")
    print("=" * 60)
    print(f"  Samples evaluated : {len(qa_samples)}")
    print(f"  Avg Token-F1      : {mean(f1_scores):.4f}")
    print(f"  Exact Match       : {mean(em_scores):.4f}")
    print(f"  Avg Faithfulness  : {mean(faith_scores):.4f}")
    print(f"  Retrieval         : Hybrid{'+Reranker' if args.use_reranker else ''}")
    print(f"  Top-K chunks      : {args.top_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
