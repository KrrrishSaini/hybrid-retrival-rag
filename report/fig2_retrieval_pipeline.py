"""Generate Figure 2: Retrieval Pipeline — Four Methods with Hybrid Score Fusion."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis("off")

# Colors
C_Q = "#E3F2FD"
C_BM25 = "#FFF9C4"
C_DENSE = "#C8E6C9"
C_HYBRID = "#E1BEE7"
C_RERANK = "#FFCCBC"
C_EVAL = "#B2EBF2"
C_BORDER = "#37474F"
C_ARROW = "#37474F"

def box(x, y, w, h, text, color, fontsize=9, bold=False):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=color, edgecolor=C_BORDER, linewidth=1.2)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, multialignment="center")

def arrow(x1, y1, x2, y2, label="", curved=False):
    style = "arc3,rad=0.15" if curved else None
    props = dict(arrowstyle="-|>", color=C_ARROW, lw=1.5)
    if style:
        props["connectionstyle"] = style
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=props)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, fontsize=7, color="#616161",
                ha="center", style="italic")

# Title
ax.text(7, 8.6, "Figure 2: Comparative Retrieval Pipeline — BM25, Dense, Hybrid, and Hybrid+Reranker",
        ha="center", va="center", fontsize=12, fontweight="bold")

# ─── Query + Chunk Store ───
box(0.3, 7.2, 2.0, 0.8, "User Query\nq", C_Q, 10, bold=True)
box(11.5, 7.2, 2.2, 0.8, "Chunk Store\n(371 chunks)", C_Q, 9, bold=True)

# ═══════════════════════════════════════════════════════════
# Method 1: BM25
# ═══════════════════════════════════════════════════════════
ax.text(0.3, 6.5, "Method 1: BM25", fontsize=9, fontweight="bold", color="#F57F17")
box(0.3, 5.5, 2.5, 0.8, "Tokenize Query\n(whitespace split)", C_BM25, 8)
box(3.3, 5.5, 2.5, 0.8, "BM25-Okapi\nScoring\n(TF-IDF + length norm)", C_BM25, 8)
box(6.3, 5.5, 2.0, 0.8, "Rank by\nBM25 Score", C_BM25, 8)
box(8.8, 5.5, 2.0, 0.8, "Top-K\nResults", C_BM25, 9, bold=True)

arrow(2.3, 7.2, 1.55, 6.3)
arrow(2.8, 5.9, 3.3, 5.9)
arrow(5.8, 5.9, 6.3, 5.9)
arrow(8.3, 5.9, 8.8, 5.9)

# Method 2: Dense
ax.text(0.3, 4.9, "Method 2: Dense", fontsize=9, fontweight="bold", color="#2E7D32")
box(0.3, 3.9, 2.5, 0.8, "Encode Query\n(all-MiniLM-L6-v2\n384-dim)", C_DENSE, 8)
box(3.3, 3.9, 2.5, 0.8, "Cosine Similarity\nvs. Chunk\nEmbeddings", C_DENSE, 8)
box(6.3, 3.9, 2.0, 0.8, "Rank by\nDense Score", C_DENSE, 8)
box(8.8, 3.9, 2.0, 0.8, "Top-K\nResults", C_DENSE, 9, bold=True)

arrow(2.3, 7.2, 1.55, 4.7)
arrow(2.8, 4.3, 3.3, 4.3)
arrow(5.8, 4.3, 6.3, 4.3)
arrow(8.3, 4.3, 8.8, 4.3)

# Chunk store arrows
arrow(11.5, 7.55, 5.8, 5.9, "index")
arrow(11.5, 7.55, 5.8, 4.3, "encode")

# ═══════════════════════════════════════════════════════════
# Method 3: Hybrid (BM25 + Dense Fusion)
# ═══════════════════════════════════════════════════════════
ax.text(0.3, 3.3, "Method 3: Hybrid (BM25 + Dense Fusion)", fontsize=9,
        fontweight="bold", color="#6A1B9A")

box(0.3, 2.2, 2.5, 0.8, "BM25: top-20\nDense: top-20\n(candidate pools)", C_HYBRID, 8)
box(3.3, 2.2, 2.8, 0.8, "Min-Max\nNormalize\nScores to [0, 1]", C_HYBRID, 8)
box(6.6, 2.2, 2.8, 0.8, "Score Fusion\nS = \u03b1\u00b7BM25_norm\n+ (1-\u03b1)\u00b7Dense_norm", C_HYBRID, 8)
box(9.9, 2.2, 2.0, 0.8, "Merged\nTop-K", C_HYBRID, 9, bold=True)

arrow(2.8, 2.6, 3.3, 2.6)
arrow(6.1, 2.6, 6.6, 2.6)
arrow(9.4, 2.6, 9.9, 2.6)

# Method 4: Hybrid + Reranker
ax.text(0.3, 1.7, "Method 4: Hybrid + Cross-Encoder Reranker", fontsize=9,
        fontweight="bold", color="#BF360C")

box(0.3, 0.6, 2.5, 0.8, "Hybrid Top-30\nCandidate Pool", C_RERANK, 8)
box(3.3, 0.6, 3.0, 0.8, "Cross-Encoder\n(ms-marco-MiniLM-L-6-v2)\nScore each (q, chunk) pair", C_RERANK, 8)
box(6.8, 0.6, 2.0, 0.8, "Re-rank by\nRelevance\nScore", C_RERANK, 8)
box(9.3, 0.6, 2.0, 0.8, "Top-K\nResults", C_RERANK, 9, bold=True)

arrow(2.8, 1.0, 3.3, 1.0)
arrow(6.3, 1.0, 6.8, 1.0)
arrow(8.8, 1.0, 9.3, 1.0)

# Evaluation metrics box
box(11.5, 1.3, 2.2, 1.6, "Evaluation\nMetrics\n\nRecall@K\nMRR@K\nnDCG@K", C_EVAL, 8, bold=True)
arrow(10.8, 5.9, 11.5, 2.6)
arrow(10.8, 4.3, 11.5, 2.4)
arrow(11.9, 2.2, 12.1, 2.9)
arrow(11.3, 1.0, 11.5, 1.6)

plt.tight_layout()
plt.savefig("report/fig2_retrieval_pipeline.png", dpi=200, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.close()
print("Figure 2 saved to report/fig2_retrieval_pipeline.png")
