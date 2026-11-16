"""Step 7: statistics of the human evaluation (Tables 2, 6, 9 and 10 of the paper).

Input:
    data/human_annotations/human_annotations_100.csv
    One row per instance; each {Dimension}_{Condition} cell is the NUMBER of the three
    annotators (0-3) who marked the dimension as present for that response.
    Conditions: Original, No_RAG, RAG (Qwen3-8B, k=20, empathy reranking).

Computes:
  * majority-vote (>=2 of 3) and unanimous (3 of 3) percentages
  * inter-rater agreement: positive-vote prevalence, observed pairwise agreement,
    chance-expected agreement, Krippendorff's alpha (nominal) and Gwet's AC1
  * paired Wilcoxon signed-rank tests on vote counts, Cliff's delta and paired
    bootstrap 95% confidence intervals for RAG vs. Original and RAG vs. No-RAG

Output:
    outputs/results/human_eval_majority.csv
    outputs/results/human_eval_agreement.csv
    outputs/results/human_eval_tests.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DATA_DIR, RESULTS_DIR, ensure_dirs

DIMENSIONS = ["Empathy", "Toxicity", "Sycophancy", "Relevance"]
CONDITIONS = ["Original", "No_RAG", "RAG"]
N_RATERS = 3


def agreement(counts: np.ndarray, n: int = N_RATERS) -> dict:
    """Agreement statistics for binary labels given per-item counts of positive votes."""
    counts = np.asarray(counts, dtype=float)
    total = len(counts) * n
    n_pos = counts.sum()
    n_neg = total - n_pos
    # Observed pairwise agreement.
    a_o = np.mean((counts * (counts - 1) + (n - counts) * (n - counts - 1)) / (n * (n - 1)))
    # Krippendorff's alpha (nominal, no missing values): expected disagreement from pooled values.
    d_e = 2 * n_pos * n_neg / (total * (total - 1))
    alpha = 1 - (1 - a_o) / d_e if d_e > 0 else np.nan
    # Gwet's AC1 (two categories).
    p = n_pos / total
    pe_ac1 = 2 * p * (1 - p)
    ac1 = (a_o - pe_ac1) / (1 - pe_ac1)
    return {"prevalence": 100 * p, "observed": 100 * a_o, "expected": 100 * (1 - d_e),
            "alpha": alpha, "ac1": ac1}


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    diff = x[:, None] - y[None, :]
    return float((diff > 0).mean() - (diff < 0).mean())


def bootstrap_ci(x, y, n_boot, seed):
    """Paired bootstrap: resample instance indices (pairs) with replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = [cliffs_delta(x[i], y[i]) for i in (rng.integers(0, n, size=n) for _ in range(n_boot))]
    return np.percentile(stats, [2.5, 97.5])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", default=str(DATA_DIR / "human_annotations" / "human_annotations_100.csv"))
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="Seed of each comparison's bootstrap.")
    args = parser.parse_args()

    df = pd.read_csv(args.annotations)
    ensure_dirs(RESULTS_DIR)

    majority, agree, tests = [], [], []
    for dim in DIMENSIONS:
        for cond in CONDITIONS:
            c = df[f"{dim}_{cond}"].to_numpy()
            majority.append({"dimension": dim, "condition": cond,
                             "majority_pct": 100 * np.mean(c >= 2), "unanimous_pct": 100 * np.mean(c == N_RATERS)})
            agree.append({"dimension": dim, "condition": cond, **agreement(c)})

        rag = df[f"{dim}_RAG"].to_numpy()
        for other in ["Original", "No_RAG"]:
            x = df[f"{dim}_{other}"].to_numpy()
            res = wilcoxon(rag, x)
            lo, hi = bootstrap_ci(rag, x, args.n_boot, args.seed)
            tests.append({"dimension": dim, "comparison": f"RAG vs {other}", "W": res.statistic,
                          "p_value": res.pvalue, "cliffs_delta": cliffs_delta(rag, x),
                          "ci_low": lo, "ci_high": hi})

    outputs = {"majority": majority, "agreement": agree, "tests": tests}
    for name, rows in outputs.items():
        table = pd.DataFrame(rows)
        table.to_csv(RESULTS_DIR / f"human_eval_{name}.csv", index=False)
        print(f"\n== {name} ==")
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
