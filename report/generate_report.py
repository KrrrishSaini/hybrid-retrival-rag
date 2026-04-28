"""Generate the Part-1 project report as a DOCX file."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.venv312', 'lib', 'python3.12', 'site-packages'))

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT

doc = Document()

# ── Page setup ──
for section in doc.sections:
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(12)
style.paragraph_format.line_spacing = 1.15
style.paragraph_format.space_after = Pt(6)

# ── Helper functions ──
def add_heading(text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = 'Times New Roman'
        run.font.color.rgb = RGBColor(0, 0, 0)
    return h

def add_para(text, bold=False, italic=False, align=None, space_after=6):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.bold = bold
    run.italic = italic
    if align:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    return p

def add_bullet(text, level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.clear()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    if level > 0:
        p.paragraph_format.left_indent = Cm(1.5 * level)
    return p


# ══════════════════════════════════════════════════════════════
# TITLE PAGE
# ══════════════════════════════════════════════════════════════
doc.add_paragraph()
doc.add_paragraph()

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = title.add_run("Chat-with-Policy-Docs:\nA Retrieval-Augmented Generation System\nfor Government Policy Question Answering")
run.font.name = 'Times New Roman'
run.font.size = Pt(22)
run.bold = True

doc.add_paragraph()

subtitle = doc.add_paragraph()
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = subtitle.add_run("NLP Project Report — Part 1")
run.font.name = 'Times New Roman'
run.font.size = Pt(16)
run.italic = True

doc.add_paragraph()
doc.add_paragraph()

info = doc.add_paragraph()
info.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = info.add_run("April 2026")
run.font.name = 'Times New Roman'
run.font.size = Pt(14)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════
add_heading("1. Introduction", level=1)

# 1.1 Problem Statement
add_heading("1.1 Problem Statement", level=2)
add_para(
    "Government policy documents, such as guidelines for Ayushman Bharat - Pradhan Mantri Jan Arogya Yojana (AB-PMJAY), "
    "Mid-Day Meal Scheme, and Pradhan Mantri Kaushal Vikas Yojana (PMKVY), are lengthy, complex PDF documents that often "
    "span hundreds of pages. Citizens, policy researchers, and implementing officers frequently need to locate specific "
    "information — eligibility criteria, coverage amounts, procedural timelines, or compliance requirements — buried "
    "within these dense texts. Manual searching is time-consuming and error-prone, particularly when answers span "
    "multiple sections or require cross-referencing between documents."
)
add_para(
    "This project addresses the challenge of building an automated question-answering system capable of accurately "
    "retrieving and synthesizing answers from unstructured policy PDFs. The core research question is: Can a lightweight, "
    "from-scratch Retrieval-Augmented Generation (RAG) pipeline — combining multiple retrieval strategies with a custom-trained "
    "decoder-only Transformer — deliver grounded, citation-backed answers from government policy documents?"
)

# 1.2 Background and Motivation
add_heading("1.2 Background and Motivation", level=2)
add_para(
    "Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for knowledge-grounded question answering, "
    "combining the precision of information retrieval with the fluency of neural language generation. While commercial "
    "systems rely on large pre-trained models (GPT-4, Claude, etc.) with billions of parameters, this project investigates "
    "whether a purpose-built, research-grade RAG system can be constructed entirely from open-source components and a "
    "custom Transformer language model trained from scratch on domain-specific data."
)
add_para(
    "The motivation is threefold. First, policy documents in the Indian government context contain domain-specific "
    "terminology, abbreviations (AB-NHPM, SECC, FCI, HAFED), and structural patterns (numbered sections, annexures, "
    "tabular data) that generic retrieval models may not handle well. Second, building the entire pipeline from scratch "
    "— from PDF extraction to language model training — provides deep insight into the strengths and limitations of each "
    "component. Third, the system demonstrates that meaningful RAG capabilities can be achieved with a compact model "
    "(approximately 20 million parameters) without requiring cloud-scale compute, making it practical for educational "
    "and resource-constrained deployments."
)

# 1.3 Objectives
add_heading("1.3 Objectives", level=2)
add_para("The project pursues the following objectives:")
add_bullet("Design and implement a complete document ingestion pipeline that converts raw policy PDFs into semantically "
           "coherent, section-aware text chunks suitable for retrieval.")
add_bullet("Build and compare four retrieval strategies — BM25 (sparse), Dense (semantic), Hybrid (score fusion), "
           "and Hybrid with Cross-Encoder Reranking — to identify the optimal retrieval configuration for policy documents.")
add_bullet("Train a custom decoder-only Transformer language model from scratch using PyTorch, including a BPE tokenizer "
           "trained on the policy corpus, and fine-tune it on question-answer pairs derived from the document collection.")
add_bullet("Develop an answer grounding and citation pipeline that attributes each generated sentence to specific source "
           "chunks, computes a faithfulness score, and falls back to extractive answers when generative quality is low.")
add_bullet("Evaluate the complete system using standard retrieval metrics (Recall@K, MRR@K, nDCG@K) and answer quality "
           "metrics (Token-F1, Exact Match, Faithfulness) on a curated 597-item QA benchmark.")
add_bullet("Deliver the system as a web application with a REST API and interactive chat interface for real-time policy "
           "document querying.")

# 1.4 Significance
add_heading("1.4 Significance", level=2)
add_para(
    "This project contributes to the understanding of RAG systems in several ways. It provides a controlled experimental "
    "environment where every component — from tokenization to generation — is implemented and trainable, enabling "
    "ablation studies that are impossible with black-box commercial APIs. The comparative evaluation of four retrieval "
    "methods on Indian government policy documents fills a gap in the literature, which has primarily focused on English "
    "Wikipedia or web-scale corpora. The extractive answer fallback mechanism addresses a practical limitation of small "
    "language models: even when the generative output is incoherent, the system can still deliver accurate, directly-quoted "
    "answers from the source documents with proper citations. Finally, the lightweight architecture demonstrates that "
    "useful RAG systems can run on consumer hardware (Apple M-series laptop with MPS acceleration), broadening accessibility."
)

# 1.5 Challenges
add_heading("1.5 Challenges", level=2)
add_para("Several technical challenges were encountered and addressed during the project:")
add_bullet("PDF Extraction Quality: Government PDFs contain inconsistent formatting, scanned pages, multi-column layouts, "
           "headers/footers repeated on every page, and tables that break across pages. Robust extraction required "
           "boilerplate deduplication via frequency analysis and careful section boundary detection using regex-based "
           "heading patterns.")
add_bullet("Small Vocabulary BPE Fragmentation: The BPE tokenizer, trained on a relatively small policy corpus with "
           "a vocabulary of 5,182 tokens, fragments common English words (e.g., 'context' becomes 'con te x t'). "
           "Since the tokenizer's decoder was not configured for subword joining, a post-processing defragmentation "
           "module was necessary to reassemble words while preserving legitimate short words.")
add_bullet("Language Model Quality at Small Scale: A 20-million parameter Transformer trained for 8 epochs on "
           "domain-specific QA data produces text that, while topically relevant, often suffers from repetition, "
           "echoing of training format artifacts, and inability to perform precise reading comprehension. This "
           "motivated the extractive-first answer strategy.")
add_bullet("Cross-Scheme Retrieval Noise: When multiple policy documents are indexed together, queries about one scheme "
           "(e.g., PMJAY) may retrieve chunks from unrelated schemes (e.g., PMKVY). Document-scope filtering was "
           "implemented to restrict retrieval to a single document when specified.")
add_bullet("Signal Handling in Threaded Environments: The retrieval models' timeout mechanism used Unix signals "
           "(SIGALRM), which fail in non-main threads. This required detection of the current thread context to "
           "conditionally disable signal-based timeouts when running under the FastAPI/uvicorn web server.")

doc.add_page_break()

# ══════════════════════════════════════════════════════════════
# 2. METHODOLOGY
# ══════════════════════════════════════════════════════════════
add_heading("2. Methodology", level=1)

# 2.1 Proposed Approach
add_heading("2.1 Proposed Approach", level=2)
add_para(
    "The system follows a modular Retrieval-Augmented Generation (RAG) architecture comprising five stages: "
    "(1) document ingestion and chunking, (2) multi-method indexing, (3) hybrid retrieval with optional reranking, "
    "(4) context-augmented generation using a custom Transformer, and (5) answer grounding with citation attribution. "
    "Figure 1 presents the complete system architecture."
)

# Insert Figure 1
fig_path_1 = os.path.join(os.path.dirname(__file__), "fig1_system_architecture.png")
if os.path.exists(fig_path_1):
    doc.add_picture(fig_path_1, width=Inches(6.2))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run("Figure 1: End-to-End System Architecture of the Chat-with-Policy-Docs RAG Pipeline. "
                      "The pipeline flows from PDF ingestion (top) through indexing, query-time retrieval, "
                      "generation, and grounded answer output (bottom).")
    run.font.size = Pt(10)
    run.italic = True
    run.font.name = 'Times New Roman'

# 2.2 Tools, Techniques, and Models
add_heading("2.2 Tools, Techniques, and Models", level=2)

add_para("Document Ingestion", bold=True, space_after=2)
add_para(
    "Raw policy PDFs are processed using PyPDF for page-level text extraction. A boilerplate deduplication step "
    "identifies and removes repeated headers, footers, and page numbers by computing line frequency across all pages "
    "and filtering lines that appear on more than 40% of pages. Section boundaries are detected using regular expression "
    "patterns that match common heading formats (numbered sections like '2.3.6', chapter headers, and annexure markers). "
    "Text is then chunked into segments of 300-800 words, respecting section boundaries to preserve semantic coherence. "
    "Each chunk is assigned a deterministic chunk_id encoding the document name, section identifier, page range, and "
    "counter, ensuring stable cross-referencing. The final corpus contains 371 chunks across three policy documents."
)

add_para("Retrieval Methods", bold=True, space_after=2)
add_para(
    "Four retrieval methods are implemented and compared, as illustrated in Figure 2. "
    "(a) BM25 Baseline: Uses the Okapi BM25 algorithm (via the rank_bm25 library) with simple whitespace tokenization. "
    "BM25 excels at exact keyword matching and is robust to vocabulary gaps. "
    "(b) Dense Retrieval: Encodes queries and chunks into 384-dimensional vectors using the sentence-transformers "
    "all-MiniLM-L6-v2 model. Nearest-neighbor search is performed with FAISS (IndexFlatIP) on platforms where FAISS is "
    "available, with a scikit-learn cosine similarity fallback on macOS. Embeddings are cached with SHA-256 validation "
    "to avoid redundant computation. "
    "(c) Hybrid Retrieval: Combines BM25 and Dense scores via weighted linear fusion. Each method independently "
    "retrieves a candidate pool of 20 chunks. Scores are min-max normalized to [0, 1], then fused as: "
    "S_final = alpha * S_BM25_norm + (1 - alpha) * S_Dense_norm, where alpha is a configurable fusion weight (default 0.5). "
    "Overlapping chunks from both pools are merged, retaining the higher fused score. "
    "(d) Hybrid + Cross-Encoder Reranking: The hybrid method first generates a larger candidate pool of 30 chunks, "
    "which are then reranked using the cross-encoder/ms-marco-MiniLM-L-6-v2 cross-encoder. This model scores each "
    "(query, chunk_text) pair jointly, providing more accurate relevance estimation at the cost of increased latency."
)

# Insert Figure 2
fig_path_2 = os.path.join(os.path.dirname(__file__), "fig2_retrieval_pipeline.png")
if os.path.exists(fig_path_2):
    doc.add_picture(fig_path_2, width=Inches(6.2))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run("Figure 2: Comparative Retrieval Pipeline. Four methods are evaluated: BM25 (sparse keyword matching), "
                      "Dense (semantic embedding), Hybrid (weighted score fusion), and Hybrid + Cross-Encoder Reranker. "
                      "All methods are evaluated using Recall@K, MRR@K, and nDCG@K metrics.")
    run.font.size = Pt(10)
    run.italic = True
    run.font.name = 'Times New Roman'

add_para("Custom Language Model", bold=True, space_after=2)
add_para(
    "A decoder-only Transformer language model is implemented from scratch in PyTorch (without using pre-built "
    "Hugging Face model classes). The architecture consists of 8 decoder blocks, each containing multi-head causal "
    "self-attention (8 heads, 64 dimensions per head) and a position-wise feed-forward network with GELU activation "
    "and 4x expansion (512 to 2048 dimensions). Pre-LayerNorm is applied before each sub-layer, and dropout (p=0.1) "
    "is used throughout. Learnable position embeddings support a context window of 512 tokens. The model contains "
    "approximately 20 million trainable parameters."
)
add_para(
    "A BPE tokenizer with a vocabulary of 5,182 tokens is trained on the chunk corpus using the Hugging Face "
    "tokenizers library, with NFC normalization and whitespace pre-tokenization. The model is trained on a QA-formatted "
    "corpus of 597 question-answer pairs (derived from the policy documents) for 8 epochs using AdamW optimization "
    "(learning rate 3e-4) with gradient clipping (max_norm=1.0). Training uses a sliding-window approach with "
    "block_size=512 and stride=128, running on Apple MPS (Metal Performance Shaders). The final training loss "
    "converges to 0.217 with a perplexity of 1.24."
)

add_para("Answer Grounding and Citation", bold=True, space_after=2)
add_para(
    "Raw LLM output undergoes a multi-stage post-processing pipeline. First, training artifacts (instruction echoes, "
    "stop-string remnants) are stripped via pattern matching. BPE tokenizer fragmentation is corrected using a "
    "defragmentation algorithm that merges adjacent short tokens while preserving real short English words via a "
    "200+ word exclusion list. Each sentence in the answer is then checked against retrieved chunks using token-overlap "
    "scoring (threshold: 0.3). Grounded sentences receive inline [chunk_id] citations; ungrounded sentences are flagged. "
    "A faithfulness score is computed as the fraction of grounded sentences."
)
add_para(
    "When the generative answer is low quality, the system falls back to an extractive strategy that selects the "
    "highest-relevance sentences directly from the retrieved chunk text using TF-IDF-weighted scoring. This ensures "
    "that even when the small LLM fails to produce coherent text, the user receives accurate, directly-quoted "
    "information from the source documents."
)

# 2.3 Workflow
add_heading("2.3 Workflow Explanation", level=2)
add_para("The end-to-end workflow operates in two phases:", space_after=2)

add_para("Offline Phase (Indexing)", bold=True, space_after=2)
add_bullet("PDF documents are placed in data/raw_pdfs/ and processed by the ingestion pipeline.")
add_bullet("Text is extracted, cleaned, and chunked into 371 section-aware segments (data/chunks.jsonl).")
add_bullet("A BM25 inverted index is built in-memory from the chunk texts.")
add_bullet("Dense embeddings are computed using all-MiniLM-L6-v2 and cached to data/indexes/dense/ with FAISS indexing.")
add_bullet("The BPE tokenizer is trained on the chunk corpus (data/tokenizer/tokenizer.json).")
add_bullet("QA training examples are generated from the QA dataset by pairing each question with its gold evidence chunks "
           "in a structured prompt format (CONTEXT / QUESTION / ANSWER / <END>).")
add_bullet("The custom Transformer is trained on the QA corpus for 8 epochs (data/llm/model.pt, 117 MB).")

add_para("Online Phase (Query Processing)", bold=True, space_after=2)
add_bullet("A user submits a natural language question through the web UI or REST API.")
add_bullet("The hybrid retriever fetches candidate chunks from both BM25 and dense indexes, normalizes and fuses scores.")
add_bullet("Optionally, the cross-encoder reranker refines the ranking of the top 30 candidates.")
add_bullet("The top-K chunks are formatted into a context block and inserted into the generation prompt.")
add_bullet("The custom Transformer generates an answer autoregressively with temperature-controlled top-k sampling.")
add_bullet("The answer grounding module attributes sentences to source chunks, computes faithfulness, and applies "
           "extractive fallback if needed.")
add_bullet("The final response — including the answer, inline citations, faithfulness score, and chunk evidence — "
           "is returned via the API and rendered in the web interface.")

add_para("Evaluation", bold=True, space_after=2)
add_para(
    "System performance is measured using a curated benchmark of 597 QA pairs with gold chunk identifiers and "
    "reference answers. Retrieval quality is evaluated with Recall@5, MRR@10, and nDCG@10 across all four retrieval "
    "methods. Answer quality is assessed with Token-F1 (token-level overlap with reference answers after stopword "
    "removal), Exact Match (normalized string equality), and Faithfulness (fraction of answer sentences grounded in "
    "retrieved evidence)."
)

# ── Save ──
out_path = os.path.join(os.path.dirname(__file__), "Report_Part1_Chat_with_Policy_Docs.docx")
doc.save(out_path)
print(f"Report saved to: {out_path}")
