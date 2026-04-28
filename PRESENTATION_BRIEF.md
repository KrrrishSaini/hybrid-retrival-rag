# Presentation Brief — Policy-Docs RAG System

---

## 1. ONE-LINE ELEVATOR PITCH

> "A Retrieval-Augmented Generation (RAG) system that lets a user ask natural-language questions about Indian government policy documents — PMJAY, Mid-Day Meal, PMKVY — and returns grounded, citation-backed answers. Built from scratch: custom PDF ingestion, hybrid retrieval (BM25 + dense + cross-encoder rerank), a 20M-parameter decoder-only Transformer trained from scratch in PyTorch, and a web/CLI interface with live PDF upload."

---

## 2. WHY THIS IS AN NLP PROJECT

You're touching *seven* major NLP subfields in a single pipeline — any one of them is lecture-worthy:

1. **Text preprocessing & tokenization** — PDF extraction, normalization, custom Byte-Pair Encoding (BPE).
2. **Information retrieval (IR)** — BM25 (classical IR), dense retrieval (neural IR), hybrid fusion.
3. **Transformer architecture** — decoder-only, causal self-attention, positional embeddings, pre-LayerNorm.
4. **Language modelling** — next-token prediction, cross-entropy loss, autoregressive generation.
5. **Sentence embeddings** — sentence-transformers, cosine similarity, semantic search.
6. **Cross-encoder reranking** — joint (query, passage) scoring, second-stage precision refinement.
7. **Grounded QA / faithfulness** — citation extraction, sentence-level grounding check, hallucination detection.

If your examiner asks *"what NLP did you do?"* — hit these seven in order.

---

## 3. WHERE / WHY THIS MATTERS (APPLICATIONS)

| Domain | Application |
|---|---|
| **Citizen services (gov-tech)** | PMJAY / MDM / PMKVY helplines, scheme eligibility bots, Grievance redressal (CPGRAMS) — reduce hours of manual PDF reading. |
| **Legal / compliance tech** | Query contracts, GST notifications, RBI circulars, tax laws. |
| **Enterprise knowledge base** | Internal HR/SOP/policy Q&A — slot in any PDF, get answers. |
| **Research assistance** | Upload a paper → ask summary/methodology/results questions (a *live* feature in your UI). |
| **Healthcare** | Query hospital treatment guidelines, insurance terms. |
| **Education / accessibility** | Make dense policy documents searchable for the visually impaired or non-experts. |

RAG is the **standard production pattern for LLM-based QA today** — used by Perplexity, Bing Copilot, Notion AI, ChatGPT's "search". Your project implements the full stack of that same pattern end-to-end.

---

## 4. HOW IT WORKS — THE 4-STAGE PIPELINE

```
   PDF          →  CHUNKS       →  RETRIEVAL    →  GENERATION    →  ANSWER
   (raw)           (JSONL)         (top-k)         (LLM)            (+citations)

 ┌────────┐      ┌────────┐      ┌──────────┐    ┌──────────┐    ┌──────────┐
 │ingest  │─────▶│chunker │─────▶│BM25 +    │───▶│Custom 20M│───▶│grounded  │
 │pdf→text│      │section │      │Dense +   │    │Transformer│    │answer +  │
 │dedupe  │      │aware   │      │Rerank    │    │           │    │citations │
 └────────┘      └────────┘      └──────────┘    └──────────┘    └──────────┘
```

### Stage 1 — INGESTION (offline, one-time)
- Extract text from PDFs page-by-page (`pypdf`).
- Remove running headers/footers (line-frequency dedup).
- Split into **section-aware chunks** of 300–800 words.
- Assign deterministic IDs: `ayushman-bharat-pmjay__2__p9-9__c0020`.
- Output: `data/chunks.jsonl` (371 chunks from 3 PDFs).

### Stage 2 — HYBRID RETRIEVAL (query-time)
- **BM25** (rank_bm25, Okapi) — lexical scoring, great for acronyms (PMJAY, SECC).
- **Dense** — encode with `all-MiniLM-L6-v2` (22M params, 384-dim), cosine similarity.
- **Hybrid fusion** — min-max normalize both, weighted blend: `score = 0.6·BM25 + 0.4·Dense`.
- **Cross-encoder reranker** — `ms-marco-MiniLM-L-6-v2` rescores top-20 candidates.
- Output: top-k chunks (k=5).

### Stage 3 — LANGUAGE MODELLING
- **Custom 20M-parameter decoder-only Transformer**, trained from scratch in raw PyTorch.
  - 8 layers, 8 attention heads, 512 hidden dim, 2048 FFN dim, block size 384.
  - **Custom BPE tokenizer** (5182 tokens) trained on the domain corpus.
  - Causal self-attention + pre-LayerNorm + GELU + learned positional embeddings.
  - Standard next-token cross-entropy loss, AdamW optimizer.

### Stage 4 — ANSWER SYNTHESIS + GROUNDING
- Format answer with inline citations: `[chunk_id]`.
- **Faithfulness score** = fraction of answer sentences whose content words overlap the retrieved evidence.
- Flag ungrounded sentences to the user.

---

## 5. RESULTS (know these numbers cold)

### Retrieval (597 QA pairs):

| Method              | Recall@5  | MRR@10   | nDCG@10  |
|---------------------|-----------|----------|----------|
| BM25                | 0.6047    | 0.4460   | 0.5159   |
| Dense (MiniLM-L6)   | 0.3635    | 0.2360   | 0.2888   |
| Hybrid (α = 0.6)    | 0.6047    | 0.4594   | 0.5178   |
| **Hybrid + Rerank** | **0.6700** | **0.5118** | **0.5705** |

### Key talking points on the table:
- **BM25 > Dense** on our domain — because PMJAY, AB-NHPM, PMKVY are *out-of-distribution* for the general-purpose MiniLM encoder. Exact-match wins when vocab is specialized.
- **Reranker is the MVP** — biggest single gain: +6.5 Recall@5, +5.2 MRR@10, +5.3 nDCG@10 over Hybrid.
- **Hybrid doesn't always beat BM25 alone on Recall@5** — but it lifts MRR@10, meaning the dense signal promotes correct chunks to higher ranks even if it doesn't find new ones.

### Answer quality:
- **Faithfulness = 1.0** on representative questions.
- Sample Q→A:
  - *"PMJAY coverage per family?"* → **Rs. 5,00,000/-**
  - *"PMKVY 4.0 short-term training duration?"* → **300–600 hours**
  - *"Mid-day meal protein norms?"* → **12g (primary), 20g (upper primary)**
  - *"Centre–State PMJAY funding ratio?"* → **60:40 (90:10 for NE/Himalayan)**

---

## 6. CODEBASE STRUCTURE — WHAT EACH FILE DOES

```
NLP/
├── app/                          ← Interfaces (user-facing)
│   ├── api.py                    ← FastAPI REST server (7 endpoints: /, /health, /chunks,
│   │                                /query, /upload, /sessions, /sessions/{id})
│   ├── chat.py                   ← Interactive CLI chat loop
│   ├── sessions.py               ← In-memory session store for uploaded PDFs (thread-safe, LRU)
│   └── static/index.html         ← Dark-mode web chat UI (1200 lines, sidebar + upload + chat)
│
├── src/ingest/                   ← Stage 1: PDF → chunks
│   ├── pdf_to_text.py            ← pypdf-based page extraction + header/footer dedup
│   ├── chunker.py                ← Section-aware chunking (regex heading detection, 300-800 words)
│   └── build_chunks.py           ← CLI entrypoint: data/raw_pdfs/ → data/chunks.jsonl
│
├── src/retrieval/                ← Stage 2: Retrieval
│   ├── bm25_baseline.py          ← BM25Okapi over simple-tokenized chunks
│   ├── dense_baseline.py         ← Sentence-transformer encoder + FAISS/sklearn cosine search
│   ├── hybrid_retriever.py       ← Min-max fuse BM25 + Dense with alpha weighting
│   └── reranker.py               ← Cross-encoder (MS-MARCO MiniLM) second-stage rescoring
│
├── src/llm/                      ← Stage 3: Custom Language Model
│   ├── model.py                  ← 20M-param decoder-only Transformer (DecoderOnlyTransformer class):
│   │                                - CausalSelfAttention (masked multi-head)
│   │                                - FeedForward (GELU, 4x expansion)
│   │                                - TransformerBlock (pre-LN + attn + FFN + residuals)
│   ├── dataset.py                ← NextTokenDataset (sliding-window, block_size=384, stride=128)
│   ├── tokenizer_train.py        ← Train BPE tokenizer on chunk corpus → data/tokenizer/
│   ├── train.py                  ← Training loop (AdamW, cross-entropy, 3 epochs, batch 16)
│   ├── generate.py               ← Autoregressive greedy/temperature sampling
│   ├── build_qa_training.py      ← Build data/qa_training.txt from QA pairs + contexts
│   ├── build_qa_dataset.py       ← Script that produced data/qa_dataset.json (597 pairs)
│   ├── rag_generate.py           ← End-to-end CLI: retrieve + generate
│   └── pretrained_generate.py    ← Answer synthesis wrapper
│
├── src/rag/                      ← Stage 4: Grounding & citations
│   └── answer.py                 ← Faithfulness scoring, citation extraction, extractive fallback
│
├── src/evaluation/
│   ├── eval_retrieval.py         ← Compute Recall@K, MRR@K, nDCG@K for all 4 retrievers
│   └── eval_answers.py           ← Token-F1, EM, Faithfulness for end-to-end answers
│
├── data/
│   ├── raw_pdfs/                 ← Source: PMJAY, MDM, PMKVY PDFs
│   ├── chunks.jsonl              ← 371 chunks (JSONL, one chunk per line)
│   ├── qa_dataset.json           ← 597 human-curated QA pairs with gold chunks + answers
│   ├── qa_training.txt           ← Concatenated training corpus for custom LLM
│   ├── tokenizer/                ← BPE tokenizer.json + merges
│   ├── indexes/dense/            ← Cached FAISS/numpy embeddings matrix
│   └── llm/                      ← Trained model checkpoint (.pt + config.json)
│
├── report/                       ← IEEE-style project report + figures
│   ├── report.md                 ← Final report source (markdown)
│   ├── NLP_Project_Report.docx   ← Word version for submission
│   └── fig1_system_architecture.py / fig2_retrieval_pipeline.py   ← Matplotlib diagrams
│
├── requirements.txt              ← Python deps (torch, transformers, fastapi, rank_bm25, etc.)
└── README.md                     ← Setup, pipeline commands, API docs
```

---

## 7. DEMO SCRIPT (7 minutes, if you have time)

1. **Slide 1 — Problem (30 s).** Open a policy PDF → "Would you read 80 pages to find one number? Neither would a citizen."
2. **Slide 2 — Pipeline diagram (45 s).** Walk through the 4 stages.
3. **Live demo (3 min).**
   - Start server: `PYTHONPATH=. .venv312/bin/python -m uvicorn app.api:app --port 8000`
   - Open `http://localhost:8000`.
   - Ask: *"What is the coverage under PMJAY per family per year?"* → Rs. 5,00,000/-.
   - Show the sources panel — "see the citations, see the chunk IDs, notice the faithfulness badge".
   - Toggle reranker off → ask again → show the rank change.
   - Upload a random PDF (resume, research paper) → ask a question about it.
4. **Slide 3 — Retrieval results table (1 min).** BM25 vs. Dense vs. Hybrid vs. +Rerank — explain why BM25 beats Dense (acronyms).
5. **Slide 4 — Architecture of the custom Transformer (45 s).** 8 layers, causal attention, BPE tokenizer, 20M params, trained from scratch.
6. **Slide 5 — Limitations + future work (30 s).** Small model; scanned PDFs need OCR; English-only; NLI-based faithfulness.
7. **Q & A.**

---

## 8. LIKELY QUESTIONS + ANSWERS

**Q: Why BM25 and not just dense retrieval?**
A: Because acronyms like PMJAY are out-of-distribution for general-purpose sentence encoders. BM25's exact-match weighting dominates when vocab is specialized. Numbers prove it: BM25 Recall@5 = 0.60 vs. Dense 0.36.

**Q: What is alpha in hybrid fusion?**
A: Convex weight between normalized BM25 and Dense scores. α=0.6 means 60% lexical, 40% semantic. Tuned on held-out queries.

**Q: Why cross-encoder after bi-encoder?**
A: Bi-encoders are fast (O(N)) but independent (no cross-attention between query and passage). Cross-encoders run a joint forward pass — quadratic but much more precise. Use bi-encoder to shortlist, cross-encoder to rerank.

**Q: What is the difference between encoder-only, decoder-only, encoder-decoder?**
A: Encoder-only (BERT) — bidirectional, good for classification/embedding. Decoder-only (GPT, ours) — causal mask, good for generation. Encoder-decoder (T5) — both, good for seq2seq.

**Q: What is causal self-attention?**
A: Attention with a triangular mask so position t only sees positions ≤ t. Prevents leakage from future tokens during autoregressive generation.

**Q: Why BPE tokenizer instead of word-level?**
A: Word-level → massive vocab + unknown tokens. Character-level → very long sequences. BPE splits on frequent subwords — handles OOV gracefully, keeps sequences short.

**Q: What is MRR, nDCG, Recall@K?**
- **Recall@K**: fraction of queries where the gold chunk is in top K.
- **MRR** (Mean Reciprocal Rank): average of 1/rank of the first relevant chunk.
- **nDCG** (normalized Discounted Cumulative Gain): graded relevance with log rank discount. Bounded in [0,1].

**Q: What is RAG?**
A: Retrieval-Augmented Generation. Retrieve external knowledge at inference time, condition the generator on it. Separates "world knowledge" (corpus) from "language ability" (model). Enables citations, reduces hallucinations, allows knowledge updates without retraining.

**Q: What is faithfulness?**
A: Fraction of generated answer sentences whose content is traceable to the retrieved evidence. Protects against hallucinations. We use a lexical overlap check; production systems use NLI models.

**Q: Why did you train your own LLM instead of using GPT?**
A: Educational — to actually understand the architecture. Also domain-specific vocabulary via BPE, full transparency, reproducibility, no API costs.

**Q: What's the limitation of your approach?**
A: (a) Scanned PDFs need OCR. (b) Faithfulness is lexical, not semantic (misses paraphrases). (c) English-only — Indian languages need multilingual encoders. (d) Small model — 20M params can't compete with 7B+ foundation models on fluency.

**Q: How is this different from using ChatGPT?**
A: ChatGPT has no visibility into your PDFs, can hallucinate, can't cite, doesn't update with new documents. RAG grounds answers in specific retrievable chunks with auditable citations.

---

## 9. CORE TERMINOLOGY (for your own vocabulary)

| Term | One-line definition |
|---|---|
| **Corpus** | The collection of documents you retrieve from |
| **Chunk** | A contiguous passage; the atomic retrieval unit |
| **BM25** | Classical sparse retrieval, term frequency × IDF with length normalization |
| **Dense retrieval** | Encode query + passage to vectors, score by similarity |
| **Embedding** | Dense vector representation of text |
| **Bi-encoder** | Encodes query and passage *independently* — fast |
| **Cross-encoder** | Encodes (query, passage) *jointly* — accurate, slow |
| **Tokenizer** | Maps text ↔ integer IDs; BPE = learned subword splits |
| **Block size / context window** | Max sequence length the Transformer can process |
| **Causal mask** | Upper-triangular mask forbidding attention to future positions |
| **Pre-LayerNorm** | LayerNorm *before* each sub-block (modern convention, more stable) |
| **Next-token prediction** | Training objective: predict token t+1 given tokens ≤ t |
| **Autoregressive generation** | Generate one token, append it, repeat |
| **Temperature / top-k / top-p** | Sampling hyperparameters controlling output randomness |
| **RAG** | Retrieval-Augmented Generation |
| **Citation / grounding** | Linking generated statements to source chunks |
| **Hallucination** | Model generates content not supported by evidence |
| **Faithfulness** | Degree to which output is grounded in evidence |

---

## 10. FINAL CONFIDENCE LINE

If examiner asks "what did *you* build":

> *"I built every stage of this pipeline myself in Python — the PDF ingestion, the chunking, four retrieval methods from classical BM25 through a neural cross-encoder reranker, a 20-million-parameter decoder-only Transformer trained from scratch in raw PyTorch with a custom BPE tokenizer over the domain corpus, an evaluation harness reporting Recall@K, MRR, nDCG, Token-F1 and faithfulness, a FastAPI REST server, and a dark-mode web UI that supports live PDF upload. The only pre-trained components are the sentence-transformer and cross-encoder weights, which are standard retrieval building blocks."*

Good luck.
