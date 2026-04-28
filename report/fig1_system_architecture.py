"""Generate Figure 1: End-to-End System Architecture of the Policy-Docs RAG Pipeline."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 9))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Color palette
C_INPUT = "#E3F2FD"
C_INGEST = "#FFF3E0"
C_INDEX = "#E8F5E9"
C_RETRIEVAL = "#F3E5F5"
C_LLM = "#FCE4EC"
C_OUTPUT = "#E0F7FA"
C_BORDER = "#455A64"
C_ARROW = "#37474F"

def box(x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, wrap=True,
            multialignment="center")

def arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my, label, fontsize=7, color="#616161", style="italic")

# Title
ax.text(7, 9.6, "Figure 1: End-to-End System Architecture — Chat-with-Policy-Docs RAG",
        ha="center", va="center", fontsize=13, fontweight="bold")

# ─── Row 1: Input ───
box(0.5, 8.2, 2.2, 0.9, "Policy PDF\nDocuments", C_INPUT, 10, bold=True)

# ─── Row 2: Ingestion Pipeline ───
box(3.5, 8.2, 2.2, 0.9, "PDF Text\nExtraction\n(PyPDF)", C_INGEST, 8)
box(6.2, 8.2, 2.2, 0.9, "Boilerplate\nDeduplication\n& Cleaning", C_INGEST, 8)
box(8.9, 8.2, 2.5, 0.9, "Section-Aware\nChunking\n(300-800 words)", C_INGEST, 8)

arrow(2.7, 8.65, 3.5, 8.65)
arrow(5.7, 8.65, 6.2, 8.65)
arrow(8.4, 8.65, 8.9, 8.65)

# Chunks output
box(11.8, 8.2, 1.7, 0.9, "Chunk Store\n(JSONL)\n371 chunks", C_INDEX, 8, bold=True)
arrow(11.4, 8.65, 11.8, 8.65)

# ─── Row 3: Indexing (from chunks) ───
box(0.5, 6.3, 2.5, 0.9, "BM25 Index\n(rank_bm25)", C_INDEX, 9)
box(3.5, 6.3, 3.0, 0.9, "Dense Index\n(all-MiniLM-L6-v2\n+ FAISS/sklearn)", C_INDEX, 8)
box(7.0, 6.3, 2.8, 0.9, "BPE Tokenizer\nTraining\n(vocab=5182)", C_INDEX, 8)
box(10.3, 6.3, 3.2, 0.9, "Custom LLM Training\n(8L/512d/8h Transformer\nAdamW, 8 epochs)", C_LLM, 8)

# Arrows from chunks to indexes
arrow(12.65, 8.2, 1.75, 7.2)
arrow(12.65, 8.2, 5.0, 7.2)
arrow(12.65, 8.2, 8.4, 7.2)
arrow(9.8, 6.75, 10.3, 6.75)

# ─── Row 4: Query-time pipeline ───
# User query
box(0.3, 4.2, 1.8, 0.9, "User\nQuery", C_INPUT, 10, bold=True)

# Retrieval methods
box(2.6, 4.6, 1.8, 0.7, "BM25\nScoring", "#BBDEFB", 8)
box(2.6, 3.7, 1.8, 0.7, "Dense\nEncoding", "#C8E6C9", 8)

arrow(2.1, 4.9, 2.6, 4.9)
arrow(2.1, 4.3, 2.6, 4.05)

# Fusion
box(5.0, 4.2, 2.2, 0.9, "Hybrid Score\nFusion\n\u03b1\u00b7BM25 + (1-\u03b1)\u00b7Dense", C_RETRIEVAL, 8)
arrow(4.4, 4.9, 5.0, 4.65)
arrow(4.4, 4.05, 5.0, 4.65)

# Reranker
box(7.8, 4.2, 2.2, 0.9, "Cross-Encoder\nReranker\n(ms-marco-MiniLM)", C_RETRIEVAL, 8)
arrow(7.2, 4.65, 7.8, 4.65)

# Retrieved chunks
box(10.5, 4.2, 1.5, 0.9, "Top-K\nChunks", C_INDEX, 9, bold=True)
arrow(10.0, 4.65, 10.5, 4.65)

# ─── Row 5: Generation + Grounding ───
box(0.5, 2.2, 2.5, 0.9, "Prompt\nConstruction\n(Context + Query)", C_LLM, 8)
box(3.5, 2.2, 2.5, 0.9, "Custom Mini LLM\nAutoregressive\nGeneration", C_LLM, 9, bold=True)
box(6.5, 2.2, 2.5, 0.9, "Answer Grounding\n& Citation\nInsertion", C_OUTPUT, 8)
box(9.5, 2.2, 2.0, 0.9, "Extractive\nFallback\n(TF-IDF)", C_OUTPUT, 8)

arrow(11.25, 4.2, 1.75, 3.1)
arrow(3.0, 2.65, 3.5, 2.65)
arrow(6.0, 2.65, 6.5, 2.65)
arrow(9.0, 2.65, 9.5, 2.65)

# Arrow from LLM training to generation
arrow(11.9, 6.3, 4.75, 3.1)

# ─── Row 6: Output ───
box(4.5, 0.5, 5.0, 0.9, "Grounded Answer\nwith Inline Citations, Faithfulness Score,\nChunk Evidence", C_OUTPUT, 9, bold=True)
arrow(7.75, 2.2, 7.0, 1.4)
arrow(10.5, 2.2, 7.0, 1.4)

# Legend
legend_items = [
    (C_INPUT, "Input/Query"),
    (C_INGEST, "Ingestion"),
    (C_INDEX, "Indexing"),
    (C_RETRIEVAL, "Retrieval"),
    (C_LLM, "LLM"),
    (C_OUTPUT, "Output"),
]
for i, (color, label) in enumerate(legend_items):
    x = 0.3 + i * 2.1
    rect = FancyBboxPatch((x, 0.0), 0.3, 0.25, boxstyle="round,pad=0.02",
                          facecolor=color, edgecolor=C_BORDER, linewidth=0.8)
    ax.add_patch(rect)
    ax.text(x + 0.4, 0.12, label, fontsize=7, va="center")

plt.tight_layout()
plt.savefig("report/fig1_system_architecture.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Figure 1 saved to report/fig1_system_architecture.png")
