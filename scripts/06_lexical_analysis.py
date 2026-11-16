"""Step 6: lexical analysis of the generated rewrites (Section 2.4 / Appendix D of the paper).

For every generation file it computes:
  * jaccard            word-level Jaccard similarity between the rewrite and the original
                       response being rewritten (lowercased, whitespace-tokenised sets)
  * rationale_coverage share of the original response's annotated empathy rationales
                       (Sharma et al., 2020) that appear verbatim in the rewrite
  * words_model        mean rewrite length in words (words_original: original response)
  * emotional / exploration / interpretation
                       number of NLEs (suggested_changes) that mention each dimension

Output:
    outputs/results/lexical_analysis.csv  (one row per generation file)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, RATIONALE_COLUMNS, RELEASED_GENERATIONS_DIR, RESULTS_DIR, ensure_dirs

NLE_TERMS = ["emotional", "exploration", "interpretation"]


def jaccard(a: str, b: str) -> float:
    x, y = set(str(a).lower().split()), set(str(b).lower().split())
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def rationales_of(row) -> list:
    out = []
    for col in RATIONALE_COLUMNS:
        if pd.notna(row[col]):
            out.extend(r.strip() for r in str(row[col]).split("|"))
    return [r for r in out if r]


def analyse_file(path: Path, reference: dict) -> dict:
    df = pd.read_csv(path)
    jac, len_model, len_orig, coverage = [], [], [], []
    for row in df.itertuples():
        if row.id_query not in reference:
            continue
        original, rationales = reference[row.id_query]
        rewrite = str(row.empathic_response_rewritten)
        jac.append(jaccard(rewrite, original))
        len_model.append(len(rewrite.split()))
        len_orig.append(len(str(original).split()))
        if rationales:
            hits = sum(1 for r in rationales if r.lower() in rewrite.lower())
            coverage.append(hits / len(rationales))

    nles = df["suggested_changes"].fillna("").str.lower()
    name = path.stem
    return {
        "file": name,
        "condition": "No-RAG" if ("no_ret" in name or name.endswith("_no_rag")) else "RAG",
        "n": len(jac),
        "jaccard": np.mean(jac),
        "rationale_coverage": np.mean(coverage) if coverage else 0.0,
        "words_model": np.mean(len_model),
        "words_original": np.mean(len_orig),
        **{term: int(nles.str.contains(term).sum()) for term in NLE_TERMS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations_dir", default=str(RELEASED_GENERATIONS_DIR),
                        help="Default: the released generations; use outputs/generations for your own runs.")
    parser.add_argument("--evaluation_csv", default=str(PROCESSED_DIR / "evaluation.csv"))
    parser.add_argument("--output", default=str(RESULTS_DIR / "lexical_analysis.csv"))
    args = parser.parse_args()

    evaluation = pd.read_csv(args.evaluation_csv)
    reference = {row["id_query"]: (row["response_post"], rationales_of(row)) for _, row in evaluation.iterrows()}

    files = sorted(Path(args.generations_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No generation files found in {args.generations_dir}")
    results = pd.DataFrame([analyse_file(f, reference) for f in files])

    ensure_dirs(Path(args.output).parent)
    results.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
