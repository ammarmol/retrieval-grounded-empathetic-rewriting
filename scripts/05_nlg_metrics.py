"""Step 5: fidelity and diversity metrics of the rewrites (Table 1 and the full NLG table of the paper).

Every rewrite is compared with the original response it rewrites:
  * bleu         NLTK sentence BLEU (default 4-gram weights, no smoothing), whitespace tokens
  * meteor       NLTK METEOR on whitespace tokens
  * rouge1/rougeL  ROUGE recall (rouge-score, Porter stemmer)
  * bertscore_f1 BERTScore F1 with bert-base-uncased
  * sas          Semantic Answer Similarity: cosine similarity of
                 roberta-large-nli-stsb-mean-tokens sentence embeddings
  * distinct1    Distinct-1 of the rewrites, averaged over responses
                 (unique unigrams / tokens per response)

Example:
    python scripts/05_nlg_metrics.py
    python scripts/05_nlg_metrics.py --metrics bleu meteor rouge distinct   # no GPU needed

Output:
    outputs/results/nlg_metrics.csv  (one row per generation file)
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from common import PROCESSED_DIR, RELEASED_GENERATIONS_DIR, RESULTS_DIR, ensure_dirs

ALL_METRICS = ["bleu", "meteor", "rouge", "bertscore", "sas", "distinct"]


def distinct_n(tokens, n=1):
    if not tokens:
        return 0.0
    grams = list(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / len(tokens)


class Scorers:
    """Loads each metric backend only if requested."""

    def __init__(self, metrics, device=None):
        self.metrics = metrics
        if "bleu" in metrics or "meteor" in metrics:
            import nltk
            nltk.download("wordnet", quiet=True)
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.translate.meteor_score import meteor_score
            self.sentence_bleu, self.meteor_score = sentence_bleu, meteor_score
        if "rouge" in metrics:
            from rouge_score import rouge_scorer
            self.rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        if "bertscore" in metrics:
            from bert_score import BERTScorer
            self.bert = BERTScorer(model_type="bert-base-uncased", device=device)
        if "sas" in metrics:
            from sentence_transformers import SentenceTransformer
            self.sas_model = SentenceTransformer("roberta-large-nli-stsb-mean-tokens", device=device)

    def score_file(self, path: Path, originals: pd.Series) -> dict:
        df = pd.read_csv(path)
        if "original_response" not in df.columns:  # released files contain model outputs only
            df["original_response"] = df["id_query"].map(originals)
        df = df.dropna(subset=["empathic_response_rewritten", "original_response"])
        refs = df["original_response"].astype(str).tolist()
        hyps = df["empathic_response_rewritten"].astype(str).tolist()
        out = {"file": path.stem, "n": len(df)}

        if "bleu" in self.metrics:
            with warnings.catch_warnings():  # NLTK warns on every pair without 4-gram overlap
                warnings.simplefilter("ignore")
                out["bleu"] = np.mean([self.sentence_bleu([r.split()], h.split()) for r, h in zip(refs, hyps)])
        if "meteor" in self.metrics:
            out["meteor"] = np.mean([self.meteor_score([r.split()], h.split()) for r, h in zip(refs, hyps)])
        if "rouge" in self.metrics:
            scores = [self.rouge.score(r, h) for r, h in zip(refs, hyps)]
            out["rouge1"] = np.mean([s["rouge1"].recall for s in scores])
            out["rougeL"] = np.mean([s["rougeL"].recall for s in scores])
        if "bertscore" in self.metrics:
            _, _, f1 = self.bert.score(hyps, refs)
            out["bertscore_f1"] = f1.mean().item()
        if "sas" in self.metrics:
            emb_r = self.sas_model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
            emb_h = self.sas_model.encode(hyps, convert_to_numpy=True, normalize_embeddings=True)
            out["sas"] = float(np.mean(np.sum(emb_r * emb_h, axis=1)))
        if "distinct" in self.metrics:
            out["distinct1"] = np.mean([distinct_n(h.split(), 1) for h in hyps])
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generations_dir", default=str(RELEASED_GENERATIONS_DIR),
                        help="Default: the released generations; use outputs/generations for your own runs.")
    parser.add_argument("--evaluation_csv", default=str(PROCESSED_DIR / "evaluation.csv"))
    parser.add_argument("--metrics", nargs="+", default=ALL_METRICS, choices=ALL_METRICS)
    parser.add_argument("--device", default=None, help="e.g. cuda, cpu (default: automatic)")
    parser.add_argument("--output", default=str(RESULTS_DIR / "nlg_metrics.csv"))
    args = parser.parse_args()

    files = sorted(Path(args.generations_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No generation files found in {args.generations_dir}")

    originals = pd.read_csv(args.evaluation_csv).set_index("id_query")["response_post"]
    scorers = Scorers(args.metrics, args.device)
    rows = []
    for f in files:
        print(f"Scoring {f.name}")
        rows.append(scorers.score_file(f, originals))
    results = pd.DataFrame(rows)

    ensure_dirs(Path(args.output).parent)
    results.to_csv(args.output, index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
