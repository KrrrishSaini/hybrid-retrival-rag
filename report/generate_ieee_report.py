"""Generate IEEE conference-style report DOCX."""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.venv312', 'lib', 'python3.12', 'site-packages'))

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import copy

doc = Document()

# ── Page setup ──
for section in doc.sections:
    section.top_margin = Cm(1.9)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(1.7)
    section.right_margin = Cm(1.7)
    section.page_width = Cm(21.59)
    section.page_height = Cm(27.94)

# ── Styles ──
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(10)
style.paragraph_format.space_before = Pt(0)
style.paragraph_format.space_after = Pt(3)
style.paragraph_format.line_spacing = 1.0

# ── Helper functions ──
def add_centered(text, size=10, bold=False, italic=False, space_after=2, space_before=0):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    return p

def add_body(text, space_after=3, indent_first=True, justify=True):
    p = doc.add_paragraph()
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if indent_first:
        p.paragraph_format.first_line_indent = Cm(0.5)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    return p

def add_section_heading(text, level=1):
    """IEEE-style section headings: I. TITLE (centered, italic for subsections)."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    if level == 1:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
    else:
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
        run.italic = True
    return p

def add_bullet(text, label=""):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.left_indent = Cm(0.5)
    if label:
        run = p.add_run(label)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.bold = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)
    return p

def add_fig_caption(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.space_before = Pt(2)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9)
    return p

def enable_columns(section, num_cols=2, spacing=Cm(0.6)):
    """Enable two-column layout for a section."""
    sectPr = section._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = parse_xml(f'<w:cols {nsdecls("w")} w:num="{num_cols}" w:space="{int(spacing)}"/>')
        sectPr.append(cols)
    else:
        cols.set(qn('w:num'), str(num_cols))
        cols.set(qn('w:space'), str(int(spacing)))


# ══════════════════════════════════════════════════════════════
# TITLE (single column)
# ══════════════════════════════════════════════════════════════

add_centered(
    "Chat-with-Policy-Docs: A Retrieval-Augmented\nGeneration System for Government Policy\nQuestion Answering",
    size=18, bold=True, space_after=10, space_before=6
)

# Author block
add_centered("Krish Saini", size=11, bold=True, space_after=1)
add_centered("School of Engineering and Technology", size=9, italic=True, space_after=1)
add_centered("BML Munjal University", size=9, italic=True, space_after=1)
add_centered("Gurugram, India", size=9, italic=True, space_after=1)
add_centered("230708", size=9, space_after=6)

# Mentor
add_centered("Mentored by: Dr. Atul Mishra", size=9, italic=True, space_after=10)

# ── Abstract ──
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_before = Pt(6)
p.paragraph_format.space_after = Pt(4)
run = p.add_run("Abstract\u2014 ")
run.font.name = 'Times New Roman'
run.font.size = Pt(10)
run.bold = True
run.italic = True
run = p.add_run(
    "This paper presents a complete Retrieval-Augmented Generation (RAG) system for question answering "
    "over Indian government policy documents. The system ingests PDF documents, performs section-aware "
    "chunking, and indexes the resulting 371 text segments using four retrieval strategies: BM25, dense "
    "semantic retrieval (all-MiniLM-L6-v2), hybrid score fusion, and hybrid with cross-encoder reranking "
    "(ms-marco-MiniLM-L-6-v2). A custom 20-million parameter decoder-only Transformer language model is "
    "trained from scratch in PyTorch on 597 domain-specific question\u2013answer pairs. Generated answers are "
    "grounded against source chunks via token-overlap scoring, producing inline citations and a faithfulness "
    "metric, with an extractive fallback mechanism for low-quality generative outputs. The system is deployed "
    "as a FastAPI web application with an interactive chat interface. Retrieval is evaluated using Recall@K, "
    "MRR@K, and nDCG@K; answer quality is measured with Token-F1, Exact Match, and Faithfulness."
)
run.font.name = 'Times New Roman'
run.font.size = Pt(10)

# Keywords
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(8)
run = p.add_run("Keywords\u2014 ")
run.font.name = 'Times New Roman'
run.font.size = Pt(10)
run.bold = True
run.italic = True
run = p.add_run(
    "Retrieval-Augmented Generation, BM25, Dense Retrieval, Hybrid Retrieval, Cross-Encoder Reranking, "
    "Decoder-only Transformer, BPE Tokenization, Answer Grounding, Policy Documents"
)
run.font.name = 'Times New Roman'
run.font.size = Pt(10)
run.italic = True

# ══════════════════════════════════════════════════════════════
# I. INTRODUCTION
# ══════════════════════════════════════════════════════════════
add_section_heading("I. INTRODUCTION")

add_body(
    "Government policy documents, such as guidelines for Ayushman Bharat \u2013 Pradhan Mantri Jan Arogya "
    "Yojana (AB-PMJAY), Mid-Day Meal Scheme, and Pradhan Mantri Kaushal Vikas Yojana (PMKVY), are "
    "lengthy, complex PDF documents that often span hundreds of pages. Citizens, policy researchers, and "
    "implementing officers frequently need to locate specific information \u2014 eligibility criteria, coverage "
    "amounts, procedural timelines, or compliance requirements \u2014 buried within these dense texts. Manual "
    "searching is time-consuming and error-prone, particularly when answers span multiple sections or "
    "require cross-referencing between documents."
)

add_body(
    "Retrieval-Augmented Generation (RAG) has emerged as the dominant paradigm for knowledge-grounded "
    "question answering, combining the precision of information retrieval with the fluency of neural "
    "language generation [1]. While commercial systems rely on large pre-trained models with billions of "
    "parameters, this project investigates whether a purpose-built, research-grade RAG system can be "
    "constructed entirely from open-source components and a custom Transformer language model trained "
    "from scratch on domain-specific data."
)

add_body(
    "The core research question is: Can a lightweight, from-scratch RAG pipeline \u2014 combining multiple "
    "retrieval strategies with a custom-trained decoder-only Transformer \u2014 deliver grounded, "
    "citation-backed answers from government policy documents? The motivation is threefold: (1) policy "
    "documents contain domain-specific terminology and structural patterns that generic retrieval models "
    "may not handle well; (2) building the pipeline from scratch provides deep insight into the strengths "
    "and limitations of each component; and (3) meaningful RAG capabilities can be achieved with a compact "
    "model (~20M parameters) on consumer hardware (Apple M-series with MPS acceleration)."
)

add_section_heading("A. Objectives", level=2)

add_body("The project pursues the following objectives:", indent_first=False, space_after=1)
add_bullet("Design a document ingestion pipeline converting raw policy PDFs into section-aware text chunks.", "\u2022 ")
add_bullet("Build and compare four retrieval strategies: BM25, Dense, Hybrid, and Hybrid+Reranker.", "\u2022 ")
add_bullet("Train a custom decoder-only Transformer LLM from scratch using PyTorch, including a BPE tokenizer.", "\u2022 ")
add_bullet("Develop answer grounding with inline citations, faithfulness scoring, and extractive fallback.", "\u2022 ")
add_bullet("Evaluate using Recall@K, MRR@K, nDCG@K, Token-F1, Exact Match, and Faithfulness on a 597-item QA benchmark.", "\u2022 ")
add_bullet("Deploy as a web application with REST API and interactive chat interface.", "\u2022 ")

add_section_heading("B. Challenges", level=2)

add_body(
    "Several technical challenges were addressed. Government PDFs contain inconsistent formatting, "
    "scanned pages, repeated headers/footers, and tables spanning multiple pages, requiring boilerplate "
    "deduplication via frequency analysis and regex-based section detection. The BPE tokenizer, trained on "
    "a small corpus (vocab=5,182), fragments common words (e.g., \u2018context\u2019 \u2192 \u2018con te x t\u2019), "
    "necessitating a post-processing defragmentation module. The 20M-parameter Transformer, while topically "
    "relevant, produces repetitive text with training format artifacts, motivating the extractive-first "
    "answer strategy. Cross-scheme retrieval noise was mitigated via document-scope filtering, and "
    "signal-based timeouts (SIGALRM) were adapted for threaded environments under FastAPI/uvicorn."
)

# ══════════════════════════════════════════════════════════════
# II. METHODOLOGY
# ══════════════════════════════════════════════════════════════
add_section_heading("II. METHODOLOGY")

add_body(
    "The system follows a modular RAG architecture comprising five stages: (1) document ingestion and "
    "chunking, (2) multi-method indexing, (3) hybrid retrieval with optional reranking, (4) context-augmented "
    "generation using a custom Transformer, and (5) answer grounding with citation attribution. Fig. 1 "
    "presents the complete system architecture.",
    indent_first=True
)

# Figure 1
fig1_path = os.path.join(os.path.dirname(__file__), "fig1_system_architecture.png")
if os.path.exists(fig1_path):
    doc.add_picture(fig1_path, width=Inches(6.5))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_fig_caption("Fig. 1. End-to-end system architecture of the Chat-with-Policy-Docs RAG pipeline. "
                    "The flow proceeds from PDF ingestion (top) through indexing, query-time retrieval, "
                    "generation, and grounded answer output (bottom).")

add_section_heading("A. Document Ingestion", level=2)

add_body(
    "Raw policy PDFs are processed using PyPDF for page-level text extraction. A boilerplate deduplication "
    "step removes repeated headers, footers, and page numbers by computing line frequency across all pages "
    "and filtering lines appearing on more than 40% of pages. Section boundaries are detected using regular "
    "expression patterns matching numbered headings (e.g., \u20182.3.6\u2019), chapter headers, and annexure markers. "
    "Text is chunked into segments of 300\u2013800 words respecting section boundaries to preserve semantic coherence. "
    "Each chunk receives a deterministic chunk_id encoding the document name, section identifier, page range, "
    "and counter. The final corpus contains 371 chunks across three policy documents."
)

add_section_heading("B. Retrieval Methods", level=2)

add_body(
    "Four retrieval methods are implemented and compared, as illustrated in Fig. 2.", indent_first=True, space_after=2
)

add_body(
    "1) BM25 Baseline: Uses the Okapi BM25 algorithm with whitespace tokenization. BM25 excels at "
    "exact keyword matching and is robust to vocabulary gaps.",
    indent_first=False, space_after=2
)

add_body(
    "2) Dense Retrieval: Encodes queries and chunks into 384-dimensional vectors using "
    "sentence-transformers/all-MiniLM-L6-v2. Nearest-neighbor search uses FAISS (IndexFlatIP) with "
    "a scikit-learn cosine similarity fallback on macOS. Embeddings are cached with SHA-256 validation.",
    indent_first=False, space_after=2
)

add_body(
    "3) Hybrid Retrieval: Combines BM25 and Dense scores via weighted linear fusion. Each method "
    "independently retrieves 20 candidate chunks. Scores are min-max normalized to [0, 1], then fused "
    "using equation (1):",
    indent_first=False, space_after=2
)

# Equation
eq = doc.add_paragraph()
eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
eq.paragraph_format.space_after = Pt(4)
eq.paragraph_format.space_before = Pt(2)
run = eq.add_run("S_final = \u03b1 \u00b7 S_BM25_norm + (1 \u2212 \u03b1) \u00b7 S_Dense_norm       (1)")
run.font.name = 'Times New Roman'
run.font.size = Pt(10)
run.italic = True

add_body(
    "where \u03b1 is a configurable fusion weight (default 0.5). Overlapping chunks from both pools "
    "are merged, retaining the higher fused score.",
    indent_first=False, space_after=2
)

add_body(
    "4) Hybrid + Cross-Encoder Reranking: The hybrid method first generates 30 candidate chunks, "
    "which are then reranked by cross-encoder/ms-marco-MiniLM-L-6-v2. This model scores each "
    "(query, chunk_text) pair jointly for more accurate relevance estimation.",
    indent_first=False, space_after=4
)

# Figure 2
fig2_path = os.path.join(os.path.dirname(__file__), "fig2_retrieval_pipeline.png")
if os.path.exists(fig2_path):
    doc.add_picture(fig2_path, width=Inches(6.5))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_fig_caption("Fig. 2. Comparative retrieval pipeline showing the four methods: BM25, Dense, "
                    "Hybrid (score fusion), and Hybrid + Cross-Encoder Reranker with evaluation metrics.")

add_section_heading("C. Custom Language Model", level=2)

add_body(
    "A decoder-only Transformer language model is implemented from scratch in PyTorch without "
    "pre-built Hugging Face model classes. The architecture, detailed in Table I, consists of 8 decoder "
    "blocks with multi-head causal self-attention (8 heads, 64 dimensions per head) and position-wise "
    "feed-forward networks with GELU activation and 4\u00d7 expansion (512 \u2192 2048). Pre-LayerNorm is applied "
    "before each sub-layer with dropout (p=0.1) throughout. Learnable position embeddings support a "
    "512-token context window. The model contains approximately 20 million trainable parameters."
)

# Table I: Model Configuration
add_centered("TABLE I", size=9, bold=True, space_before=6, space_after=1)
add_centered("CUSTOM TRANSFORMER MODEL CONFIGURATION", size=9, bold=True, space_after=4)

table = doc.add_table(rows=9, cols=2, style='Table Grid')
table.alignment = WD_ALIGN_PARAGRAPH.CENTER
headers = [("Parameter", "Value")]
rows_data = [
    ("Vocabulary Size", "5,182 tokens"),
    ("Embedding Dimension", "512"),
    ("Decoder Blocks", "8"),
    ("Attention Heads", "8 (64 dim/head)"),
    ("Feed-Forward Expansion", "4\u00d7 (512 \u2192 2,048)"),
    ("Context Window", "512 tokens"),
    ("Dropout", "0.1"),
    ("Total Parameters", "~20 million"),
]

for i, (col1, col2) in enumerate(headers + rows_data):
    for j, val in enumerate([col1, col2]):
        cell = table.cell(i, j)
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(val)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        if i == 0:
            run.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph().paragraph_format.space_after = Pt(4)

add_body(
    "A BPE tokenizer with 5,182 tokens is trained on the chunk corpus using the Hugging Face tokenizers "
    "library with NFC normalization and whitespace pre-tokenization. The model is trained on 597 "
    "QA-formatted examples for 8 epochs using AdamW (lr=3\u00d710\u207b\u2074) with gradient clipping "
    "(max_norm=1.0), a sliding window (block_size=512, stride=128), running on Apple MPS. Final "
    "training loss converges to 0.217 with perplexity 1.24."
)

add_section_heading("D. Answer Grounding and Citation", level=2)

add_body(
    "Raw LLM output undergoes multi-stage post-processing. Training artifacts (instruction echoes, "
    "stop-string remnants) are stripped via pattern matching. BPE fragmentation is corrected using "
    "a defragmentation algorithm that merges adjacent short tokens while preserving real English words "
    "via a 200+ word exclusion list. Each answer sentence is checked against retrieved chunks using "
    "token-overlap scoring (threshold=0.3). Grounded sentences receive inline [chunk_id] citations; "
    "ungrounded sentences are flagged. A faithfulness score is computed as shown in equation (2):"
)

eq2 = doc.add_paragraph()
eq2.alignment = WD_ALIGN_PARAGRAPH.CENTER
eq2.paragraph_format.space_after = Pt(4)
eq2.paragraph_format.space_before = Pt(2)
run = eq2.add_run("Faithfulness = |grounded sentences| / |total sentences|       (2)")
run.font.name = 'Times New Roman'
run.font.size = Pt(10)
run.italic = True

add_body(
    "When the generative answer is low quality, the system falls back to an extractive strategy "
    "that selects the highest-relevance sentences directly from retrieved chunk text using TF-IDF-weighted "
    "scoring, ensuring accurate, directly-quoted information with proper citations."
)

add_section_heading("E. Workflow", level=2)

add_body(
    "The end-to-end workflow operates in two phases. In the offline phase, PDF documents are ingested, "
    "cleaned, and chunked into 371 segments. BM25 and FAISS dense indexes are built, a BPE tokenizer "
    "is trained, QA training examples are generated, and the custom Transformer is trained for 8 epochs "
    "(producing a 117 MB checkpoint). In the online phase, a user submits a question through the web UI "
    "or REST API. The hybrid retriever fetches candidate chunks, the optional reranker refines the "
    "ranking, retrieved chunks are formatted into a generation prompt, the LLM generates an answer "
    "autoregressively, and the grounding module attributes sentences to sources with faithfulness "
    "scoring and extractive fallback."
)

add_section_heading("F. Evaluation Metrics", level=2)

add_body(
    "System performance is measured on a curated benchmark of 597 QA pairs with gold chunk identifiers "
    "and reference answers. Retrieval quality uses Recall@5, MRR@10, and nDCG@10 across all four methods. "
    "Answer quality uses Token-F1 (token-level overlap after stopword removal), Exact Match (normalized "
    "string equality), and Faithfulness (fraction of answer sentences grounded in retrieved evidence).",
    space_after=6
)

# ══════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════
add_section_heading("REFERENCES")

refs = [
    '[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," '
    'Advances in Neural Information Processing Systems, vol. 33, pp. 9459\u20139474, 2020.',

    '[2] S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," '
    'Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333\u2013389, 2009.',

    '[3] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," '
    'Proceedings of the 2019 Conference on Empirical Methods in NLP, pp. 3982\u20133992, 2019.',

    '[4] J. Johnson, M. Douze, and H. J\u00e9gou, "Billion-scale similarity search with GPUs," '
    'IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535\u2013547, 2021.',

    '[5] A. Vaswani et al., "Attention Is All You Need," '
    'Advances in Neural Information Processing Systems, vol. 30, 2017.',

    '[6] R. Sennrich, B. Haddow, and A. Birch, "Neural Machine Translation of Rare Words with Subword Units," '
    'Proceedings of the 54th Annual Meeting of the ACL, pp. 1715\u20131725, 2016.',
]

for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.first_line_indent = Cm(-0.5)
    run = p.add_run(ref)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(9)

# ── Save ──
out_path = os.path.join(os.path.dirname(__file__), "Report_IEEE_Style.docx")
doc.save(out_path)
print(f"IEEE-style report saved to: {out_path}")
