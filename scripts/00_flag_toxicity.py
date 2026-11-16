"""Step 0 (optional): flag toxic responses in the Sharma et al. (2020) corpus.

Uses the RoBERTa toxicity classifier of Logacheva et al. (2022)
(s-nlp/roberta_toxicity_classifier); a response is toxic when the argmax class is
"toxic" (index 1). Inputs are truncated to 512 tokens.

The flags used in the paper are already released in data/splits/, so this step is only
needed to re-derive them. With --check, the new flags are compared with the released ones.

Output:
    data/processed/toxicity_flags.csv  columns: id_query, toxicity
"""

import argparse

import pandas as pd
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from common import PROCESSED_DIR, RAW_DIR, RAW_FILES, SPLITS_DIR, ensure_dirs

MODEL = "s-nlp/roberta_toxicity_classifier"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default=str(RAW_DIR))
    parser.add_argument("--check", action="store_true", help="Compare with the released flags.")
    args = parser.parse_args()

    # Responses are identical across the three raw files; one is enough.
    df = pd.read_csv(f"{args.raw_dir}/{RAW_FILES['emotional_reactions']}")
    df = df.drop_duplicates(subset=["rp_id"])
    df["id_query"] = df["sp_id"].astype(str) + "_" + df["rp_id"].astype(str)

    tokenizer = RobertaTokenizer.from_pretrained(MODEL)
    model = RobertaForSequenceClassification.from_pretrained(MODEL).eval()

    flags = []
    with torch.no_grad():
        for text in df["response_post"].astype(str):
            inputs = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
            flags.append(int(model(inputs).logits.argmax(dim=1).item() == 1))
    df["toxicity"] = flags

    ensure_dirs(PROCESSED_DIR)
    out = PROCESSED_DIR / "toxicity_flags.csv"
    df[["id_query", "toxicity"]].to_csv(out, index=False)
    print(f"{sum(flags)} toxic responses out of {len(df)}; saved to {out}")

    if args.check:
        released = pd.concat([pd.read_csv(SPLITS_DIR / f) for f in ("learning_ids.csv", "evaluation_ids.csv")])
        merged = released.merge(df[["id_query", "toxicity"]], on="id_query", suffixes=("_released", "_new"))
        agree = (merged["toxicity_released"] == merged["toxicity_new"]).mean()
        print(f"Agreement with released flags: {100 * agree:.2f}% over {len(merged)} responses")


if __name__ == "__main__":
    main()
