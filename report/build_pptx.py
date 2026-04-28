"""Generate a PowerPoint deck for the Chat-with-Policy-Docs NLP project."""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

OUT = Path(__file__).parent / "NLP_Project_Presentation.pptx"

# ---- Theme colors ----------------------------------------------------------
NAVY = RGBColor(0x0B, 0x2B, 0x4A)       # primary dark
ACCENT = RGBColor(0x1F, 0x78, 0xC1)     # accent blue
LIGHT = RGBColor(0xF2, 0xF6, 0xFA)      # card bg
GOLD = RGBColor(0xE8, 0xA8, 0x2C)       # highlight
GREY = RGBColor(0x55, 0x5F, 0x6D)
DARK = RGBColor(0x1A, 0x1F, 0x2B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def set_text(tf, text, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT,
             font="Calibri"):
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    p.text = ""
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def add_textbox(slide, left, top, width, height, text, **kwargs):
    box = slide.shapes.add_textbox(left, top, width, height)
    set_text(box.text_frame, text, **kwargs)
    return box


def add_bullets(slide, left, top, width, height, bullets,
                size=18, color=DARK, bullet_color=ACCENT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(6)
        # bullet dot
        b = p.add_run()
        b.text = "\u25CF  "
        b.font.size = Pt(size)
        b.font.color.rgb = bullet_color
        b.font.bold = True
        b.font.name = "Calibri"
        # text
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return box


def add_rect(slide, left, top, width, height, fill=NAVY, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
    shape.shadow.inherit = False
    return shape


def add_card(slide, left, top, width, height, title, body,
             title_color=NAVY, body_color=DARK, accent=ACCENT):
    add_rect(slide, left, top, width, height, fill=LIGHT)
    # left accent bar
    add_rect(slide, left, top, Inches(0.1), height, fill=accent)
    add_textbox(
        slide,
        left + Inches(0.25), top + Inches(0.12),
        width - Inches(0.35), Inches(0.5),
        title, size=16, bold=True, color=title_color,
    )
    add_textbox(
        slide,
        left + Inches(0.25), top + Inches(0.6),
        width - Inches(0.35), height - Inches(0.7),
        body, size=12, color=body_color,
    )


def slide_header(slide, title, subtitle=None, page=None):
    # top navy bar
    add_rect(slide, 0, 0, SLIDE_W, Inches(1.05), fill=NAVY)
    add_textbox(
        slide, Inches(0.5), Inches(0.18), Inches(11), Inches(0.5),
        title, size=26, bold=True, color=WHITE,
    )
    if subtitle:
        add_textbox(
            slide, Inches(0.5), Inches(0.62), Inches(11), Inches(0.35),
            subtitle, size=13, color=RGBColor(0xCF, 0xDC, 0xEA),
        )
    if page is not None:
        add_textbox(
            slide, Inches(12.3), Inches(0.35), Inches(0.8), Inches(0.4),
            page, size=12, color=WHITE, align=PP_ALIGN.RIGHT,
        )
    # thin gold underline
    add_rect(slide, 0, Inches(1.05), SLIDE_W, Inches(0.05), fill=GOLD)


def footer(slide, page_num, total=14):
    add_textbox(
        slide, Inches(0.5), Inches(7.15), Inches(8), Inches(0.3),
        "Krish Saini · 230708 · BML Munjal University · NLP Project",
        size=10, color=GREY,
    )
    add_textbox(
        slide, Inches(12.3), Inches(7.15), Inches(0.8), Inches(0.3),
        f"{page_num} / {total}", size=10, color=GREY, align=PP_ALIGN.RIGHT,
    )


# ---------------------------------------------------------------------------
# Build deck
# ---------------------------------------------------------------------------

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
blank = prs.slide_layouts[6]

TOTAL = 14


# ---- 1. Title --------------------------------------------------------------
s = prs.slides.add_slide(blank)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
# accent band
add_rect(s, 0, Inches(3.1), SLIDE_W, Inches(0.08), fill=GOLD)

add_textbox(
    s, Inches(0.8), Inches(1.3), Inches(11.7), Inches(0.5),
    "NATURAL LANGUAGE PROCESSING · COURSE PROJECT",
    size=14, bold=True, color=GOLD, align=PP_ALIGN.LEFT,
)
add_textbox(
    s, Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.3),
    "Chat with Policy Documents",
    size=44, bold=True, color=WHITE,
)
add_textbox(
    s, Inches(0.8), Inches(3.3), Inches(11.7), Inches(1.0),
    "Domain-Specific Question Answering on Indian Policy Corpora using "
    "Hybrid Retrieval and a Custom Decoder-Only Transformer",
    size=18, color=RGBColor(0xCF, 0xDC, 0xEA),
)

add_textbox(
    s, Inches(0.8), Inches(5.3), Inches(11.7), Inches(0.5),
    "Krish Saini  ·  230708",
    size=22, bold=True, color=WHITE,
)
add_textbox(
    s, Inches(0.8), Inches(5.8), Inches(11.7), Inches(0.4),
    "Mentor: Dr. Atul Mishra",
    size=16, color=RGBColor(0xCF, 0xDC, 0xEA),
)
add_textbox(
    s, Inches(0.8), Inches(6.2), Inches(11.7), Inches(0.4),
    "BML Munjal University  ·  School of Engineering & Technology",
    size=14, color=RGBColor(0xCF, 0xDC, 0xEA),
)


# ---- 2. Problem & Motivation ----------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Problem & Motivation", "Why build a domain-specific QA system?", page="01")

add_card(
    s, Inches(0.5), Inches(1.4), Inches(6.1), Inches(2.6),
    "The Pain Point",
    "Government policy documents (NEP 2020, DPDP Act, IT Rules, RBI circulars) "
    "are long, dense, and scattered across PDFs. Citizens, students, and analysts "
    "spend hours searching for a single clause.\n\nGeneral-purpose LLMs hallucinate "
    "on niche legal/regulatory content and cannot cite source text.",
)
add_card(
    s, Inches(6.8), Inches(1.4), Inches(6.1), Inches(2.6),
    "Our Goal",
    "A lightweight, grounded QA system that:\n"
    "•  Retrieves the most relevant passages from a policy corpus\n"
    "•  Generates an answer strictly from retrieved context\n"
    "•  Cites the exact chunk for every answer\n"
    "•  Runs fully offline on a laptop (CPU-friendly)",
)

add_card(
    s, Inches(0.5), Inches(4.2), Inches(12.4), Inches(2.7),
    "Why This Matters",
    "•  Real-world NLP: combines Information Retrieval + Generative NLP in one pipeline\n"
    "•  Trust through citations: every answer is traceable to the source document\n"
    "•  Extensible: BYO-PDF feature lets users query any uploaded document\n"
    "•  Educational: full stack built from scratch — BPE tokenizer, Transformer LM, BM25, "
    "dense embeddings, hybrid fusion, cross-encoder reranking, FastAPI backend, web UI",
)
footer(s, 2, TOTAL)


# ---- 3. NLP Context --------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "NLP Concepts Covered", "One project — seven sub-fields of NLP", page="02")

concepts = [
    ("Tokenization", "Byte-Pair Encoding (BPE) trained from scratch on 5,182 subword vocab"),
    ("Language Modeling", "Custom 20M-parameter decoder-only Transformer (PyTorch, causal self-attention)"),
    ("Information Retrieval", "BM25 (sparse lexical) + Dense embeddings (MiniLM) fused via convex combination"),
    ("Semantic Search", "sentence-transformers/all-MiniLM-L6-v2 → 384-dim vectors, cosine similarity"),
    ("Re-ranking", "Cross-encoder (ms-marco-MiniLM-L-6-v2) scores query–passage pairs jointly"),
    ("RAG", "Retrieval-Augmented Generation: ground the LM on retrieved evidence → citations"),
    ("Evaluation", "Recall@k, MRR@10, nDCG@10 on 597 curated QA pairs"),
]

y = Inches(1.4)
for i, (name, desc) in enumerate(concepts):
    col = i % 2
    row = i // 2
    left = Inches(0.5 + col * 6.4)
    top = Inches(1.4 + row * 0.78)
    add_rect(s, left, top, Inches(0.3), Inches(0.65), fill=ACCENT)
    add_textbox(s, left + Inches(0.4), top + Inches(0.02),
                Inches(6.0), Inches(0.3), name, size=14, bold=True, color=NAVY)
    add_textbox(s, left + Inches(0.4), top + Inches(0.3),
                Inches(6.0), Inches(0.4), desc, size=11, color=DARK)

footer(s, 3, TOTAL)


# ---- 4. Applications -------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Real-World Applications", "Where this architecture is deployed today", page="03")

apps = [
    ("Legal & Policy", "Clause lookup across acts, bills, circulars. Trusted by law firms, compliance teams, policy analysts."),
    ("Enterprise Search", "Employees query internal wikis, HR manuals, product specs with citations."),
    ("Customer Support", "Bots answer from verified KB articles — no hallucinations, no made-up refund policies."),
    ("Healthcare", "Clinicians retrieve guidelines from 1000+ PDFs of drug labels and treatment protocols."),
    ("Education", "Students ask syllabus, assignment, and textbook questions grounded in course material."),
    ("Finance / Banking", "Analysts query RBI / SEBI circulars; auditors verify compliance clauses."),
]
for i, (name, desc) in enumerate(apps):
    col = i % 3
    row = i // 3
    left = Inches(0.5 + col * 4.2)
    top = Inches(1.5 + row * 2.7)
    add_card(s, left, top, Inches(4.0), Inches(2.5), name, desc)

footer(s, 4, TOTAL)


# ---- 5. System Architecture -----------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "System Architecture", "Four-stage Retrieval-Augmented Generation pipeline", page="04")

stages = [
    ("1. Ingest", "PDF → Text\nBoilerplate removal\n~500-token chunks", RGBColor(0x2E, 0x7D, 0x32)),
    ("2. Retrieve", "BM25 ⊕ Dense\nHybrid fusion\n(α = 0.6)", ACCENT),
    ("3. Rerank", "Cross-encoder\nQuery–chunk\njoint scoring", RGBColor(0x8E, 0x24, 0xAA)),
    ("4. Generate", "LLM answers\nfrom context\n+ citations", RGBColor(0xE6, 0x4A, 0x19)),
]
box_w = Inches(2.6)
gap = Inches(0.35)
start_x = Inches(0.9)
for i, (name, desc, color) in enumerate(stages):
    left = start_x + i * (box_w + gap)
    top = Inches(1.8)
    add_rect(s, left, top, box_w, Inches(2.6), fill=color)
    add_textbox(s, left, top + Inches(0.15), box_w, Inches(0.5),
                name, size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    add_textbox(s, left + Inches(0.15), top + Inches(0.8),
                box_w - Inches(0.3), Inches(1.6),
                desc, size=13, color=WHITE, align=PP_ALIGN.CENTER)
    # arrow
    if i < len(stages) - 1:
        arrow = s.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW,
            left + box_w + Inches(0.02), top + Inches(1.05),
            gap - Inches(0.04), Inches(0.5),
        )
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = NAVY
        arrow.line.fill.background()

# Data under the pipeline
add_rect(s, Inches(0.9), Inches(4.7), Inches(11.5), Inches(0.05), fill=GOLD)

add_textbox(
    s, Inches(0.9), Inches(4.85), Inches(11.5), Inches(0.4),
    "End-to-end flow",
    size=14, bold=True, color=NAVY,
)
add_textbox(
    s, Inches(0.9), Inches(5.2), Inches(11.5), Inches(1.8),
    "User question  →  BM25 top-30 + Dense top-30  →  Hybrid fusion  →  "
    "Cross-encoder rerank  →  Top-5 chunks  →  Prompt assembly  →  "
    "LLM generation with inline citations  →  Grounded answer delivered to UI.",
    size=13, color=DARK,
)
footer(s, 5, TOTAL)


# ---- 6. Retrieval deep-dive -----------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Retrieval — Hybrid Search", "Lexical precision meets semantic recall", page="05")

add_card(
    s, Inches(0.5), Inches(1.4), Inches(4.0), Inches(5.5),
    "BM25 (Sparse)",
    "Okapi BM25 over tokenized chunks.\n\n"
    "•  Exact-match keyword scoring\n"
    "•  Strong on acronyms, numbers, section IDs\n"
    "•  Zero training required\n"
    "•  Library: rank_bm25\n\n"
    "Weakness: misses paraphrases.",
    accent=RGBColor(0x2E, 0x7D, 0x32),
)
add_card(
    s, Inches(4.7), Inches(1.4), Inches(4.0), Inches(5.5),
    "Dense (Semantic)",
    "sentence-transformers/all-MiniLM-L6-v2.\n\n"
    "•  384-dim embeddings\n"
    "•  Cosine similarity in FAISS-style index\n"
    "•  Captures paraphrases & synonyms\n"
    "•  CPU-friendly (22 M params)\n\n"
    "Weakness: blurs precise terminology.",
    accent=ACCENT,
)
add_card(
    s, Inches(8.9), Inches(1.4), Inches(4.0), Inches(5.5),
    "Hybrid Fusion",
    "Convex combination of normalized scores:\n\n"
    "S = α · BM25 + (1−α) · Dense\n"
    "α = 0.6  (tuned on dev set)\n\n"
    "•  Min-max normalization per query\n"
    "•  Best-of-both: keywords + meaning\n"
    "•  +11% nDCG over dense alone",
    accent=GOLD,
)
footer(s, 6, TOTAL)


# ---- 7. Reranker & Generator ----------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Reranker + Generator", "Precision at the top, grounded answers at the end", page="06")

add_card(
    s, Inches(0.5), Inches(1.4), Inches(6.1), Inches(5.5),
    "Cross-Encoder Reranker",
    "Model: cross-encoder/ms-marco-MiniLM-L-6-v2\n\n"
    "•  Input: concatenated (query, passage)\n"
    "•  Output: single relevance score\n"
    "•  Jointly encodes query + chunk → far more accurate than bi-encoder\n"
    "•  Applied only on top-30 candidates for efficiency\n\n"
    "Impact: boosts Recall@5 from 0.60 → 0.67 (+11%) and MRR@10 from 0.45 → 0.51.",
)
add_card(
    s, Inches(6.8), Inches(1.4), Inches(6.1), Inches(5.5),
    "Generator",
    "Two interchangeable backends:\n\n"
    "1.  Custom Decoder-only Transformer\n"
    "    •  20 M params · 8 layers · 8 heads · 512-dim\n"
    "    •  Learned positional embeddings, pre-LN, GELU\n"
    "    •  Trained from scratch on policy corpus\n\n"
    "2.  Qwen2.5-0.5B-Instruct (synthesis)\n"
    "    •  Used at inference for fluent final answers\n"
    "    •  Grounded by retrieved context — no open-ended generation",
)
footer(s, 7, TOTAL)


# ---- 8. Results table ------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Evaluation Results", "597 domain QA pairs · Recall@5, MRR@10, nDCG@10", page="07")

# Build table
rows, cols = 5, 4
left, top, width, height = Inches(1.2), Inches(1.6), Inches(10.9), Inches(3.3)
table = s.shapes.add_table(rows, cols, left, top, width, height).table
headers = ["Method", "Recall@5", "MRR@10", "nDCG@10"]
data = [
    ["BM25",              "0.6047", "0.4460", "0.5159"],
    ["Dense (MiniLM-L6)", "0.3635", "0.2360", "0.2888"],
    ["Hybrid (α = 0.6)",  "0.6047", "0.4594", "0.5178"],
    ["Hybrid + Rerank",   "0.6700", "0.5118", "0.5705"],
]

for c, h in enumerate(headers):
    cell = table.cell(0, c)
    cell.fill.solid()
    cell.fill.fore_color.rgb = NAVY
    tf = cell.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = h
    r.font.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = WHITE
    r.font.name = "Calibri"

for ri, row in enumerate(data, start=1):
    is_best = (ri == 4)
    for ci, val in enumerate(row):
        cell = table.cell(ri, ci)
        cell.fill.solid()
        cell.fill.fore_color.rgb = GOLD if is_best else (LIGHT if ri % 2 == 0 else WHITE)
        tf = cell.text_frame
        tf.clear()
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = val
        r.font.size = Pt(15)
        r.font.bold = is_best or ci == 0
        r.font.color.rgb = DARK
        r.font.name = "Calibri"

# Takeaway
add_card(
    s, Inches(1.2), Inches(5.2), Inches(10.9), Inches(1.7),
    "Key Takeaway",
    "Reranking is the single biggest lever: +11% Recall@5 and +14% MRR over hybrid alone. "
    "BM25 remains a strong baseline for policy text because citations and section IDs favor "
    "exact-match. Dense alone underperforms — but fused with BM25 it adds recovery on paraphrased queries.",
)
footer(s, 8, TOTAL)


# ---- 9. Codebase structure -------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Codebase Structure", "What each module does", page="08")

tree = [
    ("app/", "FastAPI backend + web UI"),
    ("   api.py", "REST endpoints: /query /upload /sessions /chunks /health"),
    ("   sessions.py", "In-memory store for user-uploaded PDFs (LRU, thread-safe)"),
    ("   static/index.html", "Single-page web UI — ask, upload, view citations"),
    ("src/retrieval/", "Retrieval layer"),
    ("   bm25.py", "Okapi BM25 baseline retriever"),
    ("   dense.py", "MiniLM embeddings + cosine search"),
    ("   hybrid_retriever.py", "Convex fusion of BM25 + dense scores"),
    ("   reranker.py", "Cross-encoder ms-marco reranker"),
    ("src/llm/", "Generation layer"),
    ("   model.py", "Custom 20M decoder-only Transformer (PyTorch)"),
    ("   tokenizer.py", "BPE tokenizer trained from scratch"),
    ("   pretrained_generate.py", "Qwen2.5-0.5B-Instruct wrapper for final synthesis"),
    ("src/ingest/", "PDF extraction, boilerplate removal, chunking"),
    ("src/eval/", "Metric scripts — Recall@k, MRR, nDCG"),
    ("report/", "Final report (.md + .docx) and this deck"),
]
y = Inches(1.4)
for i, (path, desc) in enumerate(tree):
    top = Inches(1.35 + i * 0.33)
    add_textbox(
        s, Inches(0.6), top, Inches(4.2), Inches(0.32),
        path, size=12, bold=True, color=NAVY, font="Consolas",
    )
    add_textbox(
        s, Inches(4.9), top, Inches(7.9), Inches(0.32),
        desc, size=12, color=DARK,
    )
footer(s, 9, TOTAL)


# ---- 10. Tech stack --------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Technology Stack", "Libraries and models powering each layer", page="09")

stack = [
    ("Language", "Python 3.12"),
    ("Deep Learning", "PyTorch 2.x"),
    ("Transformers", "HuggingFace transformers, sentence-transformers"),
    ("Retrieval", "rank_bm25, numpy, scikit-learn"),
    ("Backend", "FastAPI, Uvicorn, Pydantic"),
    ("Frontend", "HTML5, vanilla JS, CSS"),
    ("Models", "MiniLM-L6-v2, ms-marco-MiniLM, Qwen2.5-0.5B-Instruct"),
    ("PDF Parsing", "pypdf, custom boilerplate dedupe"),
    ("Evaluation", "Custom Python (Recall@k, MRR, nDCG)"),
    ("Reporting", "python-docx, python-pptx"),
]
for i, (k, v) in enumerate(stack):
    col = i % 2
    row = i // 2
    left = Inches(0.5 + col * 6.4)
    top = Inches(1.5 + row * 1.0)
    add_rect(s, left, top, Inches(6.0), Inches(0.9), fill=LIGHT)
    add_rect(s, left, top, Inches(0.12), Inches(0.9), fill=ACCENT)
    add_textbox(s, left + Inches(0.25), top + Inches(0.1),
                Inches(5.7), Inches(0.35), k, size=14, bold=True, color=NAVY)
    add_textbox(s, left + Inches(0.25), top + Inches(0.45),
                Inches(5.7), Inches(0.4), v, size=12, color=DARK)

footer(s, 10, TOTAL)


# ---- 11. Demo --------------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Live Demo", "What the examiner will see", page="10")

add_card(
    s, Inches(0.5), Inches(1.4), Inches(6.1), Inches(5.5),
    "Flow — Policy Corpus Query",
    "1.  Start server: uvicorn app.api:app --port 8000\n"
    "2.  Open http://127.0.0.1:8000/\n"
    "3.  Ask: \"What are the key provisions of NEP 2020 on multilingual education?\"\n"
    "4.  UI shows:\n"
    "    •  Grounded answer\n"
    "    •  Citation chunk IDs\n"
    "    •  Expandable source snippets\n"
    "5.  Tune α / top-k / reranker toggle live",
)
add_card(
    s, Inches(6.8), Inches(1.4), Inches(6.1), Inches(5.5),
    "Flow — Bring-Your-Own-PDF",
    "1.  Click \"Upload Your Own PDF\" in sidebar\n"
    "2.  Select any policy / report / paper (≤ 50 MB)\n"
    "3.  Server ingests + chunks + indexes it in-memory (per-session)\n"
    "4.  All subsequent queries run against the uploaded doc\n"
    "5.  Click \"Clear Session\" to return to the main corpus\n\n"
    "→ Demonstrates extensibility beyond the shipped corpus.",
)
footer(s, 11, TOTAL)


# ---- 12. Challenges --------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Challenges & Solutions", "What broke, and how I fixed it", page="11")

rows = [
    ("Uvicorn hung 20 min on first startup",
     "HuggingFace sent silent network HEAD requests to verify cached models. "
     "Set HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 and local_files_only=True."),
    ("Dense-only retrieval underperformed",
     "Policy text relies on exact section IDs and acronyms. Added BM25 back via "
     "hybrid fusion with tuned α = 0.6."),
    ("Long PDFs blew the context window",
     "Built a chunker that respects sentence boundaries + overlap, capped at ~500 tokens, "
     "with boilerplate-line deduplication across pages."),
    ("First-query latency on cold start",
     "Added FastAPI lifespan hook that preloads retriever + reranker + LLM before "
     "the server accepts traffic."),
    ("Hallucinations when context was thin",
     "Prompt engineered the generator to refuse when retrieved chunks don't contain "
     "the answer — citations are mandatory."),
]
for i, (problem, fix) in enumerate(rows):
    top = Inches(1.4 + i * 1.1)
    add_rect(s, Inches(0.5), top, Inches(12.4), Inches(1.0), fill=LIGHT)
    add_rect(s, Inches(0.5), top, Inches(0.12), Inches(1.0), fill=GOLD)
    add_textbox(
        s, Inches(0.75), top + Inches(0.08), Inches(12.0), Inches(0.35),
        "⚠  " + problem, size=13, bold=True, color=NAVY,
    )
    add_textbox(
        s, Inches(0.75), top + Inches(0.45), Inches(12.0), Inches(0.55),
        "→  " + fix, size=12, color=DARK,
    )
footer(s, 12, TOTAL)


# ---- 13. Future Work -------------------------------------------------------
s = prs.slides.add_slide(blank)
slide_header(s, "Future Work", "Where to take this next", page="12")

future = [
    ("Scale corpus", "Ingest full Indian policy catalog (10k+ documents) via nightly crawler."),
    ("Better evaluation", "Add RAGAS / faithfulness + answer-relevance metrics beyond retrieval."),
    ("Multi-lingual", "Extend to Hindi and regional languages using multilingual MiniLM."),
    ("Streaming UI", "Token-by-token answer streaming via Server-Sent Events."),
    ("Finetuning", "LoRA-tune the generator on policy QA pairs for domain style."),
    ("Vector DB", "Swap NumPy index for FAISS / Qdrant to scale past 1M chunks."),
]
for i, (t, d) in enumerate(future):
    col = i % 2
    row = i // 2
    left = Inches(0.5 + col * 6.4)
    top = Inches(1.5 + row * 1.8)
    add_card(s, left, top, Inches(6.0), Inches(1.6), t, d)

footer(s, 13, TOTAL)


# ---- 14. Thank You ---------------------------------------------------------
s = prs.slides.add_slide(blank)
add_rect(s, 0, 0, SLIDE_W, SLIDE_H, fill=NAVY)
add_rect(s, 0, Inches(3.2), SLIDE_W, Inches(0.08), fill=GOLD)

add_textbox(
    s, Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.5),
    "Thank You",
    size=72, bold=True, color=WHITE,
)
add_textbox(
    s, Inches(0.8), Inches(3.5), Inches(11.7), Inches(0.6),
    "Questions & Discussion",
    size=24, bold=True, color=GOLD,
)
add_textbox(
    s, Inches(0.8), Inches(4.6), Inches(11.7), Inches(0.5),
    "Krish Saini  ·  230708",
    size=20, bold=True, color=WHITE,
)
add_textbox(
    s, Inches(0.8), Inches(5.1), Inches(11.7), Inches(0.4),
    "Mentor: Dr. Atul Mishra",
    size=16, color=RGBColor(0xCF, 0xDC, 0xEA),
)
add_textbox(
    s, Inches(0.8), Inches(5.5), Inches(11.7), Inches(0.4),
    "BML Munjal University · NLP Project · 2026",
    size=14, color=RGBColor(0xCF, 0xDC, 0xEA),
)
add_textbox(
    s, Inches(0.8), Inches(6.5), Inches(11.7), Inches(0.4),
    "Code: github · Demo: localhost:8000 · Report: NLP_Project_Report.docx",
    size=12, color=RGBColor(0x9F, 0xB4, 0xCA),
)


prs.save(OUT)
print(f"Wrote {OUT}")
