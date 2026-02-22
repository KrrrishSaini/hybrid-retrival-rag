# Project Agents Guide

## Goal
Build a research-grade "Chat-with-Policy-Docs" system for government policy/scheme PDFs using four retrieval pipelines:
1. BM25-only
2. Dense-only
3. Hybrid (BM25 + Dense merge)
4. Hybrid + Cross-Encoder Re-ranker

This repository is organized to support iterative development from ingestion -> retrieval -> reranking -> answer generation -> evaluation.

## Repository Structure
- `data/raw_pdfs/`: input PDFs.
- `data/extracted_text/`: optional page-wise extracted text dumps.
- `data/chunks.jsonl`: chunk store produced by ingestion.
- `data/qa_dataset.json`: manual QA/eval dataset format.
- `src/ingest/`: PDF extraction + section-aware chunking pipeline.
- `src/retrieval/`: BM25/dense/hybrid retrievers (future).
- `src/rag/`: answer generation and citation logic (future).
- `src/evaluation/`: retrieval and faithfulness evaluation (future).
- `app/`: future application/UI code.

## Coding Conventions
- Use simple, readable Python with type hints and docstrings.
- Prefer CPU-friendly defaults and configurable model choices.
- Keep scripts runnable from the command line.
- Add logging to key pipeline stages.
- Avoid over-engineering; implement incrementally.

## Chunking Rules
The ingestion pipeline must:
1. Prefer section-aware chunking using heading detection patterns (`1.`, `1.1`, `Section X`, `CHAPTER`, all-caps headings).
2. Target chunk size around 300-800 tokens (approximated by word count).
3. If a section is too long, split into subchunks while preserving section metadata.
4. Deduplicate obvious repeated boilerplate headers/footers when feasible.

Required chunk JSONL schema:
```json
{
  "doc_name": "string",
  "source_path": "string",
  "page_start": 1,
  "page_end": 2,
  "section_title": "string|null",
  "section_id": "string|null",
  "chunk_id": "string",
  "text": "string"
}
```

## QA Dataset Schema
`data/qa_dataset.json` entries must follow:
```json
{
  "id": "q001",
  "type": "factual",
  "question": "...",
  "gold_answer": "...",
  "gold_chunk_ids": ["doc__sec__p1-2__c0001"],
  "notes": "optional"
}
```
Allowed `type`: `factual`, `conditional`.

## Evaluation Plan (Future)
Track retrieval and answer quality with:
- Recall@k
- MRR
- nDCG
- Faithfulness / hallucination rate

## Answering Rule (Future RAG)
Answer strictly using retrieved evidence chunks only.
- No unsupported claims.
- Every answer sentence should be grounded in retrieved evidence.
- Citation format: append chunk IDs like `[docslug__sec__p3-4__c0012]`.

## Current Run Commands
Create chunks from PDFs:
```bash
python -m src.ingest.build_chunks --input_dir data/raw_pdfs --output_path data/chunks.jsonl
```
Quick test on first N PDFs:
```bash
python -m src.ingest.build_chunks --limit_pdfs 2
```

