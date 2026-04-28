---
title: Chat with Policy Docs
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Chat with Policy Documents

**Domain-specific Question Answering on Indian policy corpora using hybrid retrieval and a custom decoder-only Transformer.**


A grounded RAG system that answers questions over Indian policy documents (NEP 2020, DPDP Act, IT Rules, RBI circulars, etc.). Every answer cites the exact source chunk. Users can also upload any PDF and query it on the fly.

---

## Architecture

```
PDF ingest → Chunks → BM25 ⊕ Dense (α=0.6) → Cross-encoder rerank → LLM generation → Cited answer
```

| Stage      | Component                                   |
| ---------- | ------------------------------------------- |
| Tokenizer  | Custom BPE (5,182 vocab, trained from scratch) |
| Sparse IR  | Okapi BM25 (`rank_bm25`)                    |
| Dense IR   | `sentence-transformers/all-MiniLM-L6-v2`    |
| Fusion     | Min-max normalized convex combo (α = 0.6)   |
| Reranker   | `cross-encoder/ms-marco-MiniLM-L-6-v2`      |
| Generator  | Custom 20 M-param decoder-only Transformer (PyTorch) + Qwen2.5-0.5B-Instruct for synthesis |
| API        | FastAPI + Uvicorn                           |
| UI         | Single-page vanilla JS                      |

---

## Results — 597 domain QA pairs

| Method              | Recall@5  | MRR@10   | nDCG@10  |
| ------------------- | --------- | -------- | -------- |
| BM25                | 0.6047    | 0.4460   | 0.5159   |
| Dense (MiniLM-L6)   | 0.3635    | 0.2360   | 0.2888   |
| Hybrid (α = 0.6)    | 0.6047    | 0.4594   | 0.5178   |
| **Hybrid + Rerank** | **0.6700**| **0.5118**| **0.5705** |

Reranking is the single biggest lever: +11 % Recall@5, +14 % MRR over hybrid alone.

---

## Run locally

```bash
git clone https://github.com/KrrrishSaini/hybrid-retrival-rag.git
cd hybrid-retrival-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --port 8000
```

Then open <http://127.0.0.1:8000/>.

First boot downloads the three HuggingFace models (~1.5 GB).

---

## Run with Docker

```bash
docker build -t policy-rag .
docker run -p 7860:7860 policy-rag
```

This is the same image used on Hugging Face Spaces.

---

## Project structure

```
app/
├── api.py              # FastAPI endpoints (/query /upload /sessions /chunks /health)
├── sessions.py         # LRU store for user-uploaded PDFs
└── static/index.html   # Single-page web UI

src/
├── ingest/             # PDF text extraction + chunking
├── retrieval/
│   ├── bm25_baseline.py
│   ├── dense_baseline.py
│   ├── hybrid_retriever.py
│   └── reranker.py
├── llm/
│   ├── model.py            # Custom 20M decoder-only Transformer
│   ├── tokenizer.py        # BPE trainer
│   └── pretrained_generate.py  # Qwen2.5 wrapper
├── rag/
│   └── answer.py
└── evaluation/         # Recall@k, MRR, nDCG

data/
├── raw_pdfs/           # Source PDFs (gitignored)
├── chunks.jsonl        # Chunked corpus (generated)
├── indexes/            # Dense index (generated)
└── llm/model.pt        # Custom Transformer checkpoint (gitignored, 117 MB)

report/
├── report.md
├── NLP_Project_Report.docx
├── NLP_Project_Presentation.pptx
└── build_*.py          # docx / pptx generators
```

---

## API

| Method | Path                      | Purpose                                |
| ------ | ------------------------- | -------------------------------------- |
| GET    | `/health`                 | Liveness probe                          |
| GET    | `/chunks`                 | List indexed documents                  |
| POST   | `/query`                  | RAG query → grounded answer + citations |
| POST   | `/upload`                 | Upload a PDF, open a query session      |
| GET    | `/sessions`               | List active upload sessions             |
| DELETE | `/sessions/{session_id}`  | Drop a session                          |

Example:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What does NEP 2020 say about multilingual education?","top_k":5}'
```

---

## Custom Transformer checkpoint

The 117 MB `data/llm/model.pt` is **not** tracked in git (exceeds GitHub's 100 MB hard limit). To use it:

- **Retrain**: run the training script under `src/llm/`
- Or pull from a Git LFS / HF Hub mirror once published.

The shipped demo runs with **Qwen2.5-0.5B-Instruct** for fluent answer synthesis grounded on retrieved chunks.

---

## License

MIT
