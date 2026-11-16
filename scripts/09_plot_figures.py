"""Step 9: plot the figures of the paper from the reported values.

  empathy_components.pdf   Figure 2: automatic empathy components, No-RAG vs. RAG
                           (Gemma3: k=5; Qwen3: k=20; empathy reranking)
  nlg_metrics.pdf          Appendix: fidelity and diversity metrics, No-RAG vs. RAG (k=5)
  human_judgements.pdf     Appendix: share of annotator judgements in which only one of the
                           two rewrites was judged better on each dimension

Usage:
    python scripts/09_plot_figures.py [output_dir]   (default: outputs/figures)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "hatch.linewidth": 0.6, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})
INK, MUTED, GRID = "#1f1f1f", "#52514e", "#e5e5e5"
RAG_COLOR = "#2a78d6"            # RAG: solid
NORAG_COLOR = "#8a8a8a"          # No-RAG: hatched / open
COMPONENT_COLORS = ["#e34948", "#2a78d6", "#1baf7a"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / name}.pdf")


# --------------------------------------------------------------------------- Figure 2
def empathy_components():
    models = ["Gemma3 12B\n($k$=5)", "Gemma3 27B\n($k$=5)", "Qwen3 8B\n($k$=20)", "Qwen3 30B\n($k$=20)"]
    # (emotional reactions, interpretations, explorations)
    norag = [(1.36, 0.23, 0.54), (1.14, 0.15, 0.65), (1.11, 0.56, 0.41), (1.27, 0.06, 0.54)]
    rag = [(1.24, 0.35, 0.60), (1.17, 0.19, 0.67), (1.17, 0.61, 0.42), (1.24, 0.07, 0.75)]
    comp = ["Emotional reactions", "Interpretations", "Explorations"]

    fig, ax = plt.subplots(figsize=(3.3, 2.55))
    w, gap = 0.36, 0.04
    for i in range(4):
        for j, (vals, cond) in enumerate([(norag[i], "No-RAG"), (rag[i], "RAG")]):
            x = i + (j - 0.5) * (w + gap)
            bottom = 0
            for v, c in zip(vals, COMPONENT_COLORS):
                if cond == "No-RAG":
                    ax.bar(x, v, w, bottom=bottom, color=c, alpha=0.35, edgecolor=c, hatch="////", linewidth=0.6)
                else:
                    ax.bar(x, v, w, bottom=bottom, color=c, edgecolor="white", linewidth=0.6)
                if v >= 0.3:
                    ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                            color=INK if cond == "No-RAG" else "white")
                bottom += v
            ax.text(x, bottom + 0.03, f"{sum(vals):.2f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
            ax.text(x, -0.06, cond, ha="center", va="top", fontsize=5.6, color=MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=6.6)
    ax.tick_params(axis="x", length=0, pad=11)
    ax.set_ylabel("Automatic empathy score (0–6)", fontsize=7)
    ax.set_ylim(0, 2.45)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=c, label=l) for c, l in zip(COMPONENT_COLORS, comp)]
    handles += [Patch(facecolor="#bbbbbb", alpha=0.5, hatch="////", edgecolor="#888888", label="No-RAG (hatched)"),
                Patch(facecolor="#555555", label="RAG (solid)")]
    ax.legend(handles=handles, ncol=3, fontsize=5.8, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), handlelength=1.4, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    save(fig, "empathy_components")


# --------------------------------------------------------------------------- NLG metrics (appendix)
def nlg_metrics():
    metrics = ["BLEU", "METEOR", "ROUGE-L", "BERTScore-F1", "SAS", "Distinct-1"]
    data = {  # (No-RAG, RAG k=5), from the full NLG results table
        "Gemma3 12B": ([0.047, 0.323, 0.502, 0.530, 0.603, 0.805], [0.093, 0.402, 0.581, 0.576, 0.647, 0.830]),
        "Gemma3 27B": ([0.069, 0.373, 0.535, 0.567, 0.650, 0.835], [0.095, 0.401, 0.553, 0.585, 0.667, 0.847]),
        "Qwen3 8B": ([0.054, 0.221, 0.307, 0.507, 0.534, 0.884], [0.234, 0.441, 0.537, 0.619, 0.655, 0.875]),
        "Qwen3 30B": ([0.011, 0.198, 0.342, 0.468, 0.498, 0.803], [0.020, 0.220, 0.383, 0.477, 0.517, 0.793]),
    }
    fig, axes = plt.subplots(1, 4, figsize=(6.3, 1.9), sharey=True)
    y = list(range(len(metrics)))[::-1]
    for ax, (model, (base, rag)) in zip(axes, data.items()):
        for yi, b, r in zip(y, base, rag):
            ax.plot([b, r], [yi, yi], color="#c9c9c9", linewidth=1.6, zorder=1, solid_capstyle="round")
        ax.scatter(base, y, s=22, facecolor="white", edgecolor=NORAG_COLOR, linewidth=1.1, zorder=2,
                   label="No-RAG", clip_on=False)
        ax.scatter(rag, y, s=22, color=RAG_COLOR, zorder=3, label="RAG ($k$=5)", clip_on=False)
        ax.set_title(model, fontsize=7.5, color=INK)
        ax.set_xlim(-0.06, 1.06)
        ax.set_xticks([0, 0.5, 1])
        ax.tick_params(labelsize=6.5, length=2)
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(metrics, fontsize=6.8)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.35, -0.2), ncol=2, frameon=False, fontsize=6.8)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    save(fig, "nlg_metrics")


# --------------------------------------------------------------------------- Human judgements (appendix)
def human_judgements():
    dims = ["Empathy", "Toxicity", "Sycophancy", "Relevance"]
    # % of non-blank annotator judgements in which only one rewrite was judged better:
    # Empathy/Relevance: only that rewrite received the label; Toxicity/Sycophancy: only the
    # OTHER rewrite received the label.
    rag_better = [9.69, 4.14, 19.40, 3.42]
    norag_better = [5.19, 4.83, 18.66, 3.08]
    same = [85.12, 91.03, 61.94, 93.49]
    n = [289, 145, 134, 292]

    fig, ax = plt.subplots(figsize=(3.3, 1.75))
    y = list(range(len(dims)))[::-1]
    ax.barh(y, rag_better, height=0.55, color=RAG_COLOR, label="RAG judged better")
    ax.barh(y, [-v for v in norag_better], height=0.55, color="#d9d9d9", edgecolor=NORAG_COLOR,
            hatch="////", linewidth=0.6, label="No-RAG judged better")
    for yi, r, nr, s, k in zip(y, rag_better, norag_better, same, n):
        ax.text(r + 0.6, yi, f"{r:.1f}", va="center", fontsize=6.3, color=INK)
        ax.text(-nr - 0.6, yi, f"{nr:.1f}", va="center", ha="right", fontsize=6.3, color=INK)
        ax.text(1.03, yi, f"{s:.0f}%   {k}", va="center", ha="left", fontsize=6, color=MUTED,
                transform=ax.get_yaxis_transform())
    ax.text(1.03, len(dims) - 0.45, "same   n", va="center", ha="left", fontsize=6, color=MUTED,
            fontweight="bold", transform=ax.get_yaxis_transform())
    ax.axvline(0, color=INK, linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(dims, fontsize=7)
    ax.set_xlim(-25, 25)
    ax.set_xticks([-20, -10, 0, 10, 20])
    ax.set_xticklabels(["20", "10", "0", "10", "20"], fontsize=6.5)
    ax.set_xlabel("% of annotator judgements", fontsize=7)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.32), ncol=2, frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.3)
    save(fig, "human_judgements")


if __name__ == "__main__":
    empathy_components()
    nlg_metrics()
    human_judgements()
