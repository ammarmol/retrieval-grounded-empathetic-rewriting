"""Step 4: generate empathetic rewrites with an LLM served through an OpenAI-compatible API (e.g. vLLM).

Each call returns a JSON object with the rewritten response and a natural language
explanation (NLE) of the suggested changes. Generation is resumable: instances already
present in the output file are skipped.

Settings used in the paper: temperature 1.0, max. 4,096 output tokens, Qwen3 thinking
mode disabled.

Examples:
    python scripts/04_generate_rewrites.py --model Qwen/Qwen3-8B --prompts prompts_colbert_top20 --rag
    python scripts/04_generate_rewrites.py --model Qwen/Qwen3-8B --prompts prompts_colbert_top20

Output (same naming as the released files in data/generations/):
    outputs/generations/{model}_no_rag.csv
    outputs/generations/{model}_k{k}_rag.csv            (prompts_colbert_top{k})
    outputs/generations/{model}_k{k}_no_rerank_rag.csv  (prompts_colbert_top{k}_no_rerank)
    columns: id_query, query, original_response, empathic_response_rewritten, suggested_changes
"""

import argparse
import json
import re

import pandas as pd
from openai import OpenAI

from common import (GENERATIONS_DIR, PROCESSED_DIR, PROMPT_TEMPLATES, PROMPTS_DIR,
                    ensure_dirs, load_config, short_model_name)

FIELDS = ["empathic_response_rewritten", "suggested_changes"]
COLUMNS = ["id_query", "query", "original_response"] + FIELDS


def parse_output(text: str) -> dict:
    """Extract the JSON fields with a tolerant regex (models do not always return valid JSON)."""
    parsed = {}
    for field in FIELDS:
        match = re.search(rf'"{field}":\s*"(.*?)"(?:,|\s*}})', text or "", re.DOTALL)
        value = match.group(1) if match else ""
        parsed[field] = value.replace("\n", " ").replace("\r", " ").replace("*", "").strip()
    return parsed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="e.g. Qwen/Qwen3-8B, google/gemma-3-12b-it")
    parser.add_argument("--prompts", required=True, help="Prompt file name in data/prompts/ (without .csv)")
    parser.add_argument("--rag", action="store_true", help="Use the RAG prompt (default: No-RAG prompt)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    config = load_config()
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"])
    model_path = (config.get("path_model") or "") + args.model

    with open(PROMPT_TEMPLATES, encoding="utf-8") as f:
        templates = json.load(f)
    system_prompt = templates["system_prompt_rag" if args.rag else "system_prompt"]
    prompt_column = "prompt_rag" if args.rag else "prompt"

    evaluation = pd.read_csv(PROCESSED_DIR / "evaluation.csv").set_index("id_query")
    prompts = pd.read_csv(PROMPTS_DIR / f"{args.prompts}.csv")

    ensure_dirs(GENERATIONS_DIR)
    if args.rag:
        # prompts_colbert_top20 -> k20 ; prompts_colbert_top3_no_rerank -> k3_no_rerank
        config_tag = re.sub(r"^prompts_colbert_top(\d+)", r"k\1", args.prompts)
        out_name = f"{short_model_name(args.model)}_{config_tag}_rag.csv"
    else:
        out_name = f"{short_model_name(args.model)}_no_rag.csv"  # the No-RAG prompt does not depend on k
    out_path = GENERATIONS_DIR / out_name

    # Resume: keep only complete rows from a previous run.
    if out_path.exists():
        done = pd.read_csv(out_path).drop_duplicates(subset=["id_query"])
        done = done[done["empathic_response_rewritten"].fillna("").str.strip() != ""]
    else:
        done = pd.DataFrame(columns=COLUMNS)
    done.to_csv(out_path, index=False)
    done_ids = set(done["id_query"])
    print(f"{len(done_ids)} instances already generated; writing to {out_path}")

    # Qwen3 models: disable thinking mode for purely instruction-following behaviour.
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen" in args.model.lower() else None

    for row in prompts.itertuples():
        if row.id_query in done_ids:
            continue
        completion = client.chat.completions.create(
            model=model_path,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": getattr(row, prompt_column)},
            ],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            extra_body=extra_body,
        )
        parsed = parse_output(completion.choices[0].message.content)
        record = {
            "id_query": row.id_query,
            "query": evaluation.loc[row.id_query, "seeker_post"],
            "original_response": evaluation.loc[row.id_query, "response_post"],
            **parsed,
        }
        pd.DataFrame([record], columns=COLUMNS).to_csv(out_path, mode="a", header=False, index=False)
        print(f"Generated {row.id_query}")


if __name__ == "__main__":
    main()
