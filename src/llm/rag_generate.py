"""RAG generation pipeline using hybrid retrieval + custom mini LLM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any

import torch

from src.retrieval.hybrid_retriever import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_DIR,
    DEFAULT_MODEL_NAME,
    HybridRetriever,
)
from src.retrieval.reranker import DEFAULT_RERANKER_MODEL, CrossEncoderReranker

from src.rag.answer import answer_with_fallback

from .generate import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    generate_text,
    load_model_and_tokenizer,
)

LOGGER = logging.getLogger(__name__)


def _snippet(text: str, max_chars: int = 200) -> str:
    """Create compact one-line snippet for CLI printing."""
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "..."


def build_context(chunks: list[dict[str, Any]], max_context_chars: int) -> str:
    """Build context string from retrieved chunks with optional truncation."""
    if not chunks:
        return ""

    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk.get("chunk_id", "NA"))
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        parts.append(f"[{idx}] chunk_id={chunk_id}\n{text}")

    context = "\n\n".join(parts).strip()
    if max_context_chars > 0 and len(context) > max_context_chars:
        return context[:max_context_chars].rstrip() + "\n...[truncated]"
    return context


def build_prompt(query: str, context: str) -> str:
    """Build prompt used for retrieval-augmented generation.

    Matches the format used during QA fine-tuning so the model recognizes
    the pattern and produces a focused answer instead of continuation.
    """
    return (
        "You are a helpful assistant. Answer ONLY using the context. "
        'If the answer is not in the context, say "Not found in context."\n\n'
        "CONTEXT:\n"
        f"{context}\n\n"
        "QUESTION:\n"
        f"{query}\n\n"
        "ANSWER:\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI args for full RAG generation pipeline."""
    parser = argparse.ArgumentParser(
        description="RAG generation with hybrid retrieval + custom mini Transformer."
    )
    parser.add_argument("--query", type=str, required=True, help="User query text.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks to use.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Hybrid fusion weight: final = alpha*bm25 + (1-alpha)*dense.",
    )
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Apply cross-encoder reranker on hybrid candidates.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only local cache for dense/reranker models (offline mode).",
    )
    parser.add_argument(
        "--top_k_hybrid_candidates",
        type=int,
        default=20,
        help="Hybrid candidate pool size before reranking.",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=6000,
        help="Max context characters before truncation.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=120, help="Max generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument(
        "--sample_top_k",
        type=int,
        default=50,
        help="Top-k cutoff for token sampling during generation.",
    )
    parser.add_argument(
        "--chunks_path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to chunks JSONL.",
    )
    parser.add_argument(
        "--index_dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to dense index cache directory.",
    )
    parser.add_argument(
        "--retriever_model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence-transformers model for dense retrieval.",
    )
    parser.add_argument(
        "--reranker_model_name",
        type=str,
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder model name for reranking.",
    )
    parser.add_argument(
        "--top_k_bm25",
        type=int,
        default=20,
        help="BM25 candidate pool size.",
    )
    parser.add_argument(
        "--top_k_dense",
        type=int,
        default=20,
        help="Dense candidate pool size.",
    )
    parser.add_argument("--retrieval_batch_size", type=int, default=64, help="Dense embedding batch size.")
    parser.add_argument("--reranker_batch_size", type=int, default=32, help="Reranker batch size.")
    parser.add_argument(
        "--retrieval_device",
        type=str,
        default="cpu",
        help="Retrieval model device (default: cpu).",
    )
    parser.add_argument(
        "--model_load_timeout_s",
        type=int,
        default=60,
        help="Timeout for dense/reranker model loading (0 disables timeout).",
    )
    parser.add_argument("--no_faiss", action="store_true", help="Skip FAISS and force sklearn backend.")
    parser.add_argument(
        "--doc_filter",
        type=str,
        default=None,
        help="Restrict retrieval to chunks whose doc_name contains this substring.",
    )
    parser.add_argument(
        "--force_rebuild_dense",
        action="store_true",
        help="Force rebuilding dense embedding cache.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to custom mini LLM checkpoint.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to custom mini LLM config JSON.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=Path,
        default=None,
        help="Optional tokenizer path override for generation.",
    )
    parser.add_argument("--llm_device", type=str, default="auto", help="LLM device: auto|cpu|mps|cuda")
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
    """CLI entrypoint for retrieval-augmented generation."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(args.seed)
    LOGGER.info("Starting RAG pipeline for query: %s", args.query)

    try:
        retriever = HybridRetriever(
            chunks_path=args.chunks_path,
            index_dir=args.index_dir,
            model_name=args.retriever_model_name,
            alpha=args.alpha,
            top_k_bm25=args.top_k_bm25,
            top_k_dense=args.top_k_dense,
            batch_size=args.retrieval_batch_size,
            device=args.retrieval_device,
            local_files_only=args.local_files_only,
            model_load_timeout_s=args.model_load_timeout_s,
            prefer_faiss=not args.no_faiss,
            force_rebuild_dense=args.force_rebuild_dense,
        )

        if args.use_reranker:
            candidate_k = max(args.top_k, args.top_k_hybrid_candidates)
            LOGGER.info("Retrieving %d hybrid candidates before reranking", candidate_k)
            candidates = retriever.retrieve(
                query=args.query, top_k=candidate_k, doc_filter=args.doc_filter
            )
            if not candidates:
                print("Error: retrieval returned no candidates.")
                return 1
            reranker = CrossEncoderReranker(
                model_name=args.reranker_model_name,
                batch_size=args.reranker_batch_size,
                device=args.retrieval_device,
                local_files_only=args.local_files_only,
                model_load_timeout_s=args.model_load_timeout_s,
            )
            results = reranker.rerank(query=args.query, candidates=candidates, top_k=args.top_k)
        else:
            results = retriever.retrieve(
                query=args.query, top_k=args.top_k, doc_filter=args.doc_filter
            )

        if not results:
            print("Error: no chunks retrieved. Check chunks file or query.")
            return 1

        context = build_context(results, max_context_chars=args.max_context_chars)
        if not context:
            print("Error: retrieved chunks had empty text. Cannot build context.")
            return 1
        prompt = build_prompt(query=args.query, context=context)

        model, tokenizer, device = load_model_and_tokenizer(
            model_path=args.model_path,
            config_path=args.config_path,
            tokenizer_path=args.tokenizer_path,
            device=args.llm_device,
        )
        output = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.sample_top_k,
            stop_strings=["<END>", "CONTEXT:", "QUESTION:"],
            device=device,
        )
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Error: {exc}")
        return 1

    # Ground the raw answer against retrieved chunks (with extractive fallback).
    grounded = answer_with_fallback(output, args.query, results)

    print("=== QUERY ===")
    print(args.query)
    print()
    print("=== TOP CHUNKS ===")
    for rank, item in enumerate(results, start=1):
        chunk_id = item.get("chunk_id", "NA")
        doc_name = item.get("doc_name") or "NA"
        section_id = item.get("section_id") or "NA"
        if args.use_reranker and "rerank_score" in item:
            score_str = (
                f"rerank={float(item.get('rerank_score', 0.0)):.4f}, "
                f"hybrid={float(item.get('final_score', 0.0)):.4f}"
            )
        else:
            score_str = f"hybrid={float(item.get('final_score', item.get('score', 0.0))):.4f}"

        print(f"{rank}. chunk_id={chunk_id} | source={doc_name}/{section_id} | {score_str}")
        print(f"   snippet={_snippet(str(item.get('text', '')))}")

    print()
    print("=== GENERATED ANSWER (with citations) ===")
    print(grounded["answer"])
    print()
    print(f"Faithfulness: {grounded['faithfulness']:.2%}")
    print(f"Citations: {', '.join(grounded['citations']) if grounded['citations'] else 'none'}")
    if grounded["ungrounded_sentences"]:
        print(f"Ungrounded sentences ({len(grounded['ungrounded_sentences'])}):")
        for sent in grounded["ungrounded_sentences"]:
            print(f"  - {sent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
