"""Step 1: build the learning and evaluation sets from the Sharma et al. (2020) corpus.

Input  (not redistributed, see data/README.md):
    data/raw/emotional-reactions-reddit.csv
    data/raw/explorations-reddit.csv
    data/raw/interpretations-reddit.csv
Input  (released):
    data/splits/learning_ids.csv, data/splits/evaluation_ids.csv
Output:
    data/processed/learning.csv     non-toxic, non-zero-empathy responses (retrieval anchors)
    data/processed/evaluation.csv   zero-empathy or toxic responses (to be rewritten)

Procedure (Section 2.2 of the paper):
  1. For each empathy dimension, drop exact duplicates and, when the same response
     (rp_id) has conflicting levels, keep the lowest level.
  2. Merge the three dimensions on rp_id (3,023 responses).
  3. Attach the toxicity flag produced by a RoBERTa toxicity classifier
     (Logacheva et al., 2022). The flag is distributed with the split files.
  4. Evaluation set = all three empathy levels are 0, OR the response is toxic.
     Learning set   = everything else.
"""

import argparse

import pandas as pd

from common import (LEVEL_COLUMNS, PROCESSED_DIR, RAW_DIR, RAW_FILES, SPLITS_DIR,
                    ensure_dirs)


def load_dimension(path, name):
    df = pd.read_csv(path)
    df = df.rename(columns={"level": f"level_{name}", "rationales": f"rationales_{name}"})
    # Remove exact duplicates (same response, same level).
    df = df.drop_duplicates(subset=["rp_id", f"level_{name}"], keep="first")
    # Conflicting annotations for the same response: keep the lowest empathy level.
    df = df.sort_values(["rp_id", f"level_{name}"]).drop_duplicates(subset=["rp_id"], keep="first")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default=str(RAW_DIR))
    args = parser.parse_args()

    frames = {name: load_dimension(f"{args.raw_dir}/{fname}", name) for name, fname in RAW_FILES.items()}

    # The three raw files contain the same posts; a few seeker posts differ only in
    # whitespace (this does not affect retrieval or results). Seeker posts are taken
    # from the interpretations file and responses from the emotional-reactions file.
    merged = frames["emotional_reactions"][["sp_id", "rp_id", "response_post"]].merge(
        frames["interpretations"][["rp_id", "seeker_post"]], on="rp_id")
    for name, df in frames.items():
        merged = merged.merge(df[["rp_id", f"level_{name}", f"rationales_{name}"]], on="rp_id")
    merged["id_query"] = merged["sp_id"].astype(str) + "_" + merged["rp_id"].astype(str)
    print(f"Merged responses: {len(merged)}")

    # Toxicity flags and split membership come from the released ID lists.
    learning_ids = pd.read_csv(SPLITS_DIR / "learning_ids.csv")
    evaluation_ids = pd.read_csv(SPLITS_DIR / "evaluation_ids.csv")
    tox = pd.concat([learning_ids, evaluation_ids])[["id_query", "toxicity"]]
    merged = merged.merge(tox, on="id_query", how="inner")

    zero_empathy = (merged[LEVEL_COLUMNS] == 0).all(axis=1)
    is_eval = zero_empathy | (merged["toxicity"] == 1)
    evaluation = merged[is_eval]
    learning = merged[~is_eval]

    # Sanity check: the rule must reproduce the released split exactly.
    assert set(evaluation["id_query"]) == set(evaluation_ids["id_query"]), "Evaluation split mismatch"
    assert set(learning["id_query"]) == set(learning_ids["id_query"]), "Learning split mismatch"

    ensure_dirs(PROCESSED_DIR)
    learning.to_csv(PROCESSED_DIR / "learning.csv", index=False)
    evaluation.to_csv(PROCESSED_DIR / "evaluation.csv", index=False)
    print(f"Learning set: {len(learning)} | Evaluation set: {len(evaluation)} "
          f"(toxic: {int(evaluation['toxicity'].sum())})")


if __name__ == "__main__":
    main()
