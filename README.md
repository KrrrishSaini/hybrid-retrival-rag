# Chat-with-Policy-Docs (Ingestion Foundation)

This repository currently implements **data ingestion + section-aware chunking** for policy PDFs. Retrieval, reranking, and answer generation modules are scaffolded for later milestones.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This installs BM25 + dense retrieval dependencies, including sentence-transformers.

## Input Data

Put source PDFs in:

- `data/raw_pdfs/`

Optional extracted page text is written to:

- `data/extracted_text/`

## Build Chunks

```bash
python -m src.ingest.build_chunks --input_dir data/raw_pdfs --output_path data/chunks.jsonl
```

Quick test (process only first N PDFs):

```bash
python -m src.ingest.build_chunks --limit_pdfs 2
```

Optional flag to write extracted text dumps:

```bash
python -m src.ingest.build_chunks --write_extracted_text
```

## BM25 Baseline Retrieval

Install dependencies:

```bash
pip install -r requirements.txt
```

Run BM25 retrieval:

```bash
python3 -m src.retrieval.bm25_baseline --query "eligibility criteria for PMJAY" --top_k 10
```

## Dense Baseline Retrieval

Run dense retrieval:

```bash
python3 -m src.retrieval.dense_baseline --query "who is eligible for PMJAY" --top_k 10
```

Another example:

```bash
python3 -m src.retrieval.dense_baseline --query "PMJAY beneficiary identification criteria" --top_k 10
```

Dense cache files are stored in:

- `data/indexes/dense/embeddings.npy`
- `data/indexes/dense/chunks_metadata.jsonl`
- `data/indexes/dense/manifest.json`

Cache rebuild behavior:
- Reuses cache when `data/chunks.jsonl` hash and model name match manifest.
- Rebuilds automatically if chunks file changes.
- Use `--force_rebuild` to force recomputation.

Mac note:
- `faiss-cpu` is optional and skipped on macOS in `requirements.txt`.
- On macOS, the retriever automatically uses `scikit-learn` (`NearestNeighbors`, cosine).
- On non-mac platforms with FAISS installed, it will prefer FAISS `IndexFlatIP`.

## Hybrid Retrieval (BM25 + Dense)

Run hybrid retrieval with weighted fusion:

```bash
python3 -m src.retrieval.hybrid_retriever --query "who is eligible for PMJAY" --top_k 5 --alpha 0.6
```

Run another example:

```bash
python3 -m src.retrieval.hybrid_retriever --query "PMJAY beneficiary identification criteria" --top_k 5 --alpha 0.5
```

Hybrid defaults:
- Candidate pools: `--top_k_bm25 20`, `--top_k_dense 20`
- Fusion score: `final = alpha * bm25_norm + (1 - alpha) * dense_norm`
- Scores are min-max normalized to `[0, 1]` before fusion.
- In restricted/offline environments, add `--local_files_only` to avoid network calls for dense model loading.

## Hybrid + Cross-Encoder Reranker

Run hybrid retrieval with reranking:

```bash
python3 -m src.retrieval.hybrid_retriever --query "who is eligible for PMJAY" --top_k 5 --alpha 0.6 --use_reranker
```

Offline/restricted-network mode:

```bash
python3 -m src.retrieval.hybrid_retriever --query "who is eligible for PMJAY" --top_k 5 --alpha 0.6 --use_reranker --local_files_only
```

How reranking works:
- Hybrid retriever first builds a candidate pool (`--top_k_hybrid_candidates`, default `20`).
- Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each `(query, chunk_text)` pair.
- Final output is sorted by `rerank_score` while still showing hybrid/BM25/dense scores for traceability.

## Retrieval Evaluation (Light)

Run retrieval evaluation on `data/qa_dataset.json`:

```bash
python3 -m src.evaluation.eval_retrieval --top_k 10 --alpha 0.6 --use_reranker
```

Metrics reported:
- `Recall@5`
- `MRR@10`

Evaluated methods:
- BM25
- Dense
- Hybrid
- Hybrid + Reranker (when `--use_reranker` is enabled)

## Output Format (`data/chunks.jsonl`)

Each line is one JSON object with fields:

- `doc_name`
- `source_path`
- `page_start`
- `page_end`
- `section_title`
- `section_id`
- `chunk_id`
- `text`

`chunk_id` is deterministic and stable for a document/chunk order.

## Current Scope

Implemented:
- PDF text extraction (page-wise)
- Boilerplate line deduplication (repeated headers/footers)
- Section-aware chunking with regex heading detection
- Chunk JSONL writer + CLI sanity output
- BM25 retrieval baseline
- Dense retrieval baseline with cache + FAISS/sklearn fallback
- Hybrid retrieval baseline (BM25 + Dense score fusion)
- Cross-encoder reranking on top of hybrid retrieval

Planned next:
- QA and retrieval evaluation suite
