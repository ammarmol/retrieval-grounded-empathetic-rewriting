# Human Examples Are Not Enough: Limits of Retrieval Grounding for Safe Empathetic Rewriting

Code, prompts, data splits, and human annotations for the EACL 2027 paper
*Human Examples Are Not Enough: Limits of Retrieval Grounding for Safe Empathetic Rewriting*.

We study whether grounding an LLM rewriter in **retrieved human-authored empathetic
responses** makes empathetic rewriting for online peer support safer than prompting
alone. Given a seeker post and a low-empathy or toxic draft written by a peer supporter,
the system retrieves similar posts with ColBERT, reranks the retrieved responses by their
human-annotated empathy score, and asks an LLM to make minimal edits to the draft and to
explain each change (a natural language explanation, NLE). The supporter then accepts,
edits, or rejects the suggestion.

## Repository structure

```
├── config.example.yaml        # inference-server settings (copy to config.yaml)
├── prompts/
│   └── rewriting_prompts.json # system and user prompts (RAG and No-RAG), as used in the paper
├── scripts/
│   ├── common.py              # shared paths and helpers
│   ├── 00_flag_toxicity.py    # (optional) RoBERTa toxicity flags, already released in data/splits/
│   ├── 01_prepare_data.py     # build learning / evaluation sets from the raw corpus
│   ├── 02_retrieve_colbert.py # ColBERT retrieval (top 20 candidates per query)
│   ├── 03_build_prompts.py    # empathy reranking and prompt construction for a given k
│   ├── 04_generate_rewrites.py# LLM rewriting through an OpenAI-compatible API (vLLM)
│   ├── 05_nlg_metrics.py      # BLEU, METEOR, ROUGE, BERTScore, SAS, Distinct-1
│   ├── 06_lexical_analysis.py # Jaccard, rationale coverage, length, NLE term counts
│   ├── 07_human_eval_stats.py # majority votes, agreement, Wilcoxon, Cliff's delta, bootstrap CIs
│   ├── 08_llm_judge.py        # LLM-as-judge validation against the human labels
│   └── 09_plot_figures.py     # Figure 2 and appendix figures
├── data/
│   ├── README.md              # how to obtain the corpus (not redistributed)
│   ├── splits/                # learning / evaluation IDs and toxicity flags
│   └── human_annotations/     # human evaluation (vote counts per instance)
└── docs/
    └── annotation_guidelines.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp config.example.yaml config.yaml # then edit base_url / api_key
```

Generation uses any OpenAI-compatible endpoint. In the paper, all models were served
locally with [vLLM](https://github.com/vllm-project/vllm), e.g.
`vllm serve Qwen/Qwen3-8B --port 8000`.

**Data.** The Reddit corpus of Sharma et al. (2020) cannot be redistributed. Follow
[`data/README.md`](data/README.md) to download it and place it in `data/raw/`.

## Reproducing the experiments

All scripts are run from the repository root.

```bash
# 0. (optional) Re-derive the toxicity flags and compare them with the released ones
python scripts/00_flag_toxicity.py --check

# 1. Learning (2,070) and evaluation (953) sets; checks against the released splits
python scripts/01_prepare_data.py

# 2. ColBERT retrieval of the 20 most similar learning-set seeker posts
python scripts/02_retrieve_colbert.py --top_k 20

# 3. Prompts: the top k candidates are reranked by empathy score (0-6) and the 3
#    highest-scoring responses are used as examples
for k in 5 10 20; do python scripts/03_build_prompts.py --k $k; done
python scripts/03_build_prompts.py --k 3 --no_rerank   # semantic-only ablation

# 4. Rewriting (temperature 1.0, max. 4,096 tokens; Qwen3 thinking mode disabled)
MODEL=Qwen/Qwen3-8B   # also: Qwen/Qwen3-30B-A3B, google/gemma-3-12b-it, google/gemma-3-27b-it
python scripts/04_generate_rewrites.py --model $MODEL --prompts prompts_colbert_top20          # No-RAG
for p in prompts_colbert_top5 prompts_colbert_top10 prompts_colbert_top20 prompts_colbert_top3_no_rerank; do
  python scripts/04_generate_rewrites.py --model $MODEL --prompts $p --rag
done

# 5. Fidelity and diversity metrics (BERTScore and SAS benefit from a GPU)
python scripts/05_nlg_metrics.py

# 6. Lexical analysis of all generations
python scripts/06_lexical_analysis.py

# 7. Human evaluation statistics (runs on the released annotations, no corpus needed)
python scripts/07_human_eval_stats.py

# 8. LLM-as-judge validation
python scripts/08_llm_judge.py --judge_model google/gemma-4-31b-it \
  --no_rag_csv outputs/generations/qwen3-8b_prompts_colbert_top20_no_rag.csv \
  --rag_csv    outputs/generations/qwen3-8b_prompts_colbert_top20_rag.csv
```

The No-RAG prompt does not depend on k, so a single No-RAG run per model is enough.

### Where each result comes from

| Result in the paper | Script |
|---|---|
| Toxicity flags used to build the splits | `00_flag_toxicity.py` |
| BLEU, METEOR, ROUGE, BERTScore-F1, SAS, Distinct-1 | `05_nlg_metrics.py` |
| Jaccard similarity and response length | `06_lexical_analysis.py` |
| Rationale coverage and NLE term frequencies | `06_lexical_analysis.py` |
| Human evaluation: majority and unanimous votes | `07_human_eval_stats.py` |
| Inter-rater agreement (prevalence, observed/expected agreement, α, AC1) | `07_human_eval_stats.py` |
| Wilcoxon tests, Cliff's δ, and bootstrap 95% CIs | `07_human_eval_stats.py` |
| LLM-as-judge agreement with human labels | `08_llm_judge.py` |
| Figures: empathy components, NLG metrics, human judgements | `09_plot_figures.py` |

<!-- TODO(authors): add the empathy classifier code before release, or keep this note. -->
### Not included in this repository

- Training and inference of the three empathy classifiers (emotional reactions,
  interpretations, explorations) used for the automatic empathy scores and Figure 2,
  which follow Sharma et al. (2020).

## Human annotations

`data/human_annotations/human_annotations_100.csv` contains, for each of the 100
evaluated instances, the number of annotators (out of 3) who marked each dimension as
present for the Original, No-RAG, and RAG responses. The guidelines are in
[`docs/annotation_guidelines.md`](docs/annotation_guidelines.md). Annotator identities are
not included.

## Citation

```bibtex
@inproceedings{marmolromero2027human,
  title     = {Human Examples Are Not Enough: Limits of Retrieval Grounding for Safe Empathetic Rewriting},
  author    = {M{\'a}rmol-Romero, Alba Mar{\'i}a and Ecevit, Isin Su and Bucur, Ana-Maria and
               Montejo-R{\'a}ez, Arturo and Crestani, Fabio},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2027},
  note      = {TODO: add pages and URL once the proceedings are published}
}
```

If you use the data, please also cite the original corpus:

```bibtex
@inproceedings{sharma2020computational,
  title     = {A Computational Approach to Understanding Empathy Expressed in Text-Based Mental Health Support},
  author    = {Sharma, Ashish and Miner, Adam and Atkins, David and Althoff, Tim},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {5263--5276},
  year      = {2020}
}
```

## License

Code, prompts, guidelines, split identifiers, and annotations: [CC BY-NC 4.0](LICENSE),
for research purposes. The Sharma et al. (2020) corpus is subject to its own terms.

## Intended use

This code is released for research on assistive, human-in-the-loop writing support.
It is not a clinical tool, does not detect crisis situations, and should not be deployed
to generate responses to people in distress without human review and the involvement
of mental health professionals.
