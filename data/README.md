# Data

The Reddit peer-support corpus of Sharma et al. (2020) **is not redistributed** in this
repository: the seeker posts and original responses are not included. This folder contains
identifiers, labels we produced, human annotations, and the model-generated rewrites.

## What is included

| Path | Content |
|---|---|
| `splits/learning_ids.csv` | 2,070 responses used as retrieval anchors (non-toxic, non-zero empathy) |
| `splits/evaluation_ids.csv` | 953 responses to be rewritten (zero empathy or toxic) |
| `generations/` | Model rewrites and explanations for all 953 evaluation instances, 20 configurations |
| `human_annotations/human_annotations_100.csv` | Human evaluation of 100 evaluation instances |

**Generations.** One file per model and configuration, named
`{model}_{config}.csv` with model in `gemma-3-12b`, `gemma-3-27b`, `qwen3-8b`, `qwen3-30b`
and config in `no_rag`, `k5_rag`, `k10_rag`, `k20_rag`, `k3_no_rerank_rag` (k is the number
of retrieved candidates; the prompt always contains the 3 highest-empathy examples among
them). Columns: `id_query`, `empathic_response_rewritten` (the rewrite) and
`suggested_changes` (the model's natural language explanation). A few rewrites are empty
where the model output could not be parsed. The rewrites annotated by humans are those of
`qwen3-8b_no_rag.csv` (No-RAG) and `qwen3-8b_k20_rag.csv` (RAG).

**Split files.** Columns: `id_query` (`{sp_id}_{rp_id}`), `sp_id` (seeker post ID),
`rp_id` (response post ID), and `toxicity` (1 if the response was flagged by the RoBERTa
toxicity classifier of Logacheva et al., 2022; 345 responses, all in the evaluation set).

**Human annotations.** One row per instance (`id_query`). Each
`{Dimension}_{Condition}` column holds the **number of annotators (0-3)** who marked
the dimension as present. Dimensions: `Empathy`, `Toxicity`, `Sycophancy`, `Relevance`.
Conditions: `Original` (the human response), `No_RAG`, and `RAG` (Qwen3-8B, k = 20, empathy
reranking). A label is positive by majority vote when the count is at least 2.
`preferred_similarity` and `preferred_naturalness` record two additional direct-preference
questions (`RAG` / `Not RAG`) that are not reported in the paper.

## What you need to download

1. Obtain the empathy corpus from the authors of Sharma et al. (2020):
   <https://github.com/behavioral-data/Empathy-Mental-Health>
2. Place the three Reddit files in `data/raw/`:
   ```
   data/raw/emotional-reactions-reddit.csv
   data/raw/explorations-reddit.csv
   data/raw/interpretations-reddit.csv
   ```
3. Run `python scripts/01_prepare_data.py`. It rebuilds `data/processed/learning.csv` and
   `data/processed/evaluation.csv` and checks that they match the released splits.

Everything produced from these files (`data/processed/`, `data/retrieval/`,
`data/prompts/`, `outputs/`) contains Reddit text and is git-ignored.

The released generations can be analysed directly with `scripts/05_nlg_metrics.py` and
`scripts/06_lexical_analysis.py` (both default to `data/generations/`), once
`data/processed/evaluation.csv` has been rebuilt with step 1.
