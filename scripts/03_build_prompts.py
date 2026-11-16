"""Step 3: build RAG and No-RAG prompts for every evaluation instance.

Retrieval depth k is the number of ColBERT candidates considered. With empathy
reranking (default), the k candidates are reranked by their human-annotated total
empathy score (0-6, sum of the three dimensions) and the top --n_examples (3) are
inserted as few-shot examples. With --no_rerank, the top --n_examples candidates
by ColBERT score are used instead.

Output:
    data/prompts/prompts_colbert_top{k}.csv            (empathy reranking)
    data/prompts/prompts_colbert_top{k}_no_rerank.csv  (--no_rerank)
    columns: id_query, prompt_rag, prompt   (prompt = No-RAG version)
"""

import argparse
import json

import pandas as pd

from common import (LEVEL_COLUMNS, PROCESSED_DIR, PROMPT_TEMPLATES, PROMPTS_DIR,
                    RETRIEVAL_DIR, ensure_dirs)


def format_examples(examples: pd.DataFrame) -> str:
    text = ""
    for i, row in enumerate(examples.itertuples(), start=1):
        text += f"Sample {i}.\n - Question: {row.seeker_post}\n - Response: {row.response_post}\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True, help="Number of retrieved candidates (5, 10 or 20).")
    parser.add_argument("--n_examples", type=int, default=3, help="Examples inserted in the prompt.")
    parser.add_argument("--no_rerank", action="store_true", help="Disable empathy-based reranking.")
    parser.add_argument("--retrieval_file", default=str(RETRIEVAL_DIR / "colbert_top20.csv"))
    args = parser.parse_args()

    with open(PROMPT_TEMPLATES, encoding="utf-8") as f:
        templates = json.load(f)

    learning = pd.read_csv(PROCESSED_DIR / "learning.csv")
    evaluation = pd.read_csv(PROCESSED_DIR / "evaluation.csv")
    retrieved = pd.read_csv(args.retrieval_file)

    # Keep the first k candidates per query (the file is ordered by ColBERT score).
    retrieved = retrieved.groupby("test_id_query", sort=False).head(args.k)
    retrieved = retrieved.merge(
        learning[["id_query", "seeker_post", "response_post"] + LEVEL_COLUMNS],
        left_on="similar_id_query", right_on="id_query", how="left", validate="many_to_one",
    )
    missing = retrieved["seeker_post"].isna().sum()
    if missing:
        raise ValueError(f"{missing} retrieved IDs were not found in the learning set.")

    if args.no_rerank:
        ranked = retrieved.sort_values(["test_id_query", "similarity"], ascending=[True, False], kind="stable")
    else:
        retrieved["empathy_score"] = retrieved[LEVEL_COLUMNS].fillna(0).sum(axis=1)
        ranked = retrieved.sort_values(["test_id_query", "empathy_score"], ascending=[True, False], kind="stable")
    selected = ranked.groupby("test_id_query", sort=False).head(args.n_examples)
    examples_by_query = {q: g for q, g in selected.groupby("test_id_query", sort=False)}

    prompts = []
    for row in evaluation.itertuples():
        examples = examples_by_query.get(row.id_query)
        if examples is None or len(examples) == 0:
            raise ValueError(f"No examples retrieved for {row.id_query}")
        prompts.append({
            "id_query": row.id_query,
            "prompt_rag": templates["prompt_rag"].format(
                seeker_post=str(row.seeker_post).strip(),
                bad_response=str(row.response_post).strip(),
                examples_with_rationales=format_examples(examples),
            ),
            "prompt": templates["prompt"].format(
                seeker_post=str(row.seeker_post).strip(),
                bad_response=str(row.response_post).strip(),
            ),
        })

    ensure_dirs(PROMPTS_DIR)
    suffix = "_no_rerank" if args.no_rerank else ""
    out = PROMPTS_DIR / f"prompts_colbert_top{args.k}{suffix}.csv"
    pd.DataFrame(prompts).to_csv(out, index=False)
    print(f"Saved {len(prompts)} prompts to {out}")


if __name__ == "__main__":
    main()
