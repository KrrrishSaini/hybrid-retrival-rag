"""Interactive CLI chat for the Policy-Docs RAG system.

Usage:
    python3 -m app.chat
    python3 -m app.chat --use_reranker --doc_filter "PMJAY"
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import torch

from src.llm.generate import load_model_and_tokenizer, generate_text
from src.llm.rag_generate import build_context, build_prompt
from src.rag.answer import answer_with_fallback
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker

LOGGER = logging.getLogger(__name__)


def _snippet(text: str, max_chars: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars].rstrip() + "..." if len(compact) > max_chars else compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive RAG chat CLI.")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--use_reranker", action="store_true")
    parser.add_argument("--doc_filter", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--llm_device", type=str, default="auto")
    parser.add_argument("--retrieval_device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--log_level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    torch.manual_seed(args.seed)

    print("Loading models... (this may take a minute)")

    try:
        retriever = HybridRetriever(
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
        model, tokenizer, llm_device = load_model_and_tokenizer(device=args.llm_device)
    except (FileNotFoundError, RuntimeError, ValueError, ImportError) as exc:
        print(f"Error loading models: {exc}")
        return 1

    doc_count = len(retriever.bm25_retriever.chunks)
    doc_names = sorted({str(c.get("doc_name", "")) for c in retriever.bm25_retriever.chunks})

    print()
    print("=" * 60)
    print("  Policy-Docs RAG Chat")
    print("=" * 60)
    print(f"  Documents: {', '.join(doc_names)}")
    print(f"  Chunks: {doc_count}")
    print(f"  Retrieval: Hybrid{'+Reranker' if reranker else ''} (alpha={args.alpha})")
    if args.doc_filter:
        print(f"  Doc filter: {args.doc_filter}")
    print()
    print("  Type your question and press Enter.")
    print("  Commands: /docs, /filter <name>, /quit")
    print("=" * 60)
    print()

    doc_filter = args.doc_filter

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return 0

        if not query:
            continue

        # Commands
        if query.lower() in ("/quit", "/exit", "/q"):
            print("Bye!")
            return 0
        if query.lower() == "/docs":
            for name in doc_names:
                print(f"  - {name}")
            continue
        if query.lower().startswith("/filter"):
            parts = query.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                doc_filter = parts[1].strip()
                print(f"  Filter set to: {doc_filter}")
            else:
                doc_filter = None
                print("  Filter cleared.")
            continue

        # Retrieve
        if reranker:
            candidates = retriever.retrieve(query=query, top_k=20, doc_filter=doc_filter)
            chunks = reranker.rerank(query=query, candidates=candidates, top_k=args.top_k)
        else:
            chunks = retriever.retrieve(query=query, top_k=args.top_k, doc_filter=doc_filter)

        if not chunks:
            print("\nBot> No relevant chunks found.\n")
            continue

        context = build_context(chunks, max_context_chars=6000)
        prompt = build_prompt(query=query, context=context)

        raw_answer = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=50,
            stop_strings=["<END>", "CONTEXT:", "QUESTION:"],
            device=llm_device,
        )

        grounded = answer_with_fallback(raw_answer, query, chunks)

        print()
        print(f"Bot> {grounded['answer']}")
        print()
        if grounded["citations"]:
            print(f"  Citations: {', '.join(grounded['citations'])}")
        print(f"  Faithfulness: {grounded['faithfulness']:.0%}")
        print(f"  Sources:")
        for i, c in enumerate(chunks, 1):
            print(f"    {i}. [{c.get('chunk_id', '')}] {_snippet(str(c.get('text', '')))}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())
