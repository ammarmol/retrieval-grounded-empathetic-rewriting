"""Shared paths and helpers for all pipeline scripts.

Every script is meant to be run from the repository root, e.g.
    python scripts/01_prepare_data.py
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # Sharma et al. (2020) files go here (not redistributed)
PROCESSED_DIR = DATA_DIR / "processed"  # learning / evaluation sets built from the raw files
SPLITS_DIR = DATA_DIR / "splits"        # released ID lists defining our splits
RETRIEVAL_DIR = DATA_DIR / "retrieval"
PROMPTS_DIR = DATA_DIR / "prompts"
RELEASED_GENERATIONS_DIR = DATA_DIR / "generations"  # model outputs used in the paper (released)
OUTPUTS_DIR = ROOT / "outputs"
GENERATIONS_DIR = OUTPUTS_DIR / "generations"
RESULTS_DIR = OUTPUTS_DIR / "results"
PROMPT_TEMPLATES = ROOT / "prompts" / "rewriting_prompts.json"
CONFIG_PATH = ROOT / "config.yaml"

# Raw files of the Sharma et al. (2020) empathy corpus, one per dimension.
RAW_FILES = {
    "emotional_reactions": "emotional-reactions-reddit.csv",
    "exploratory": "explorations-reddit.csv",
    "interpretations": "interpretations-reddit.csv",
}
LEVEL_COLUMNS = [f"level_{d}" for d in RAW_FILES]
RATIONALE_COLUMNS = [f"rationales_{d}" for d in RAW_FILES]


def load_config() -> dict:
    """Read the inference-server configuration (see config.example.yaml)."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Copy config.example.yaml to config.yaml and fill it in."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def short_model_name(model: str) -> str:
    """Map a model identifier to the short name used in output file names."""
    name = model.lower().split("/")[-1]
    for key in ("gemma-3-12b", "gemma-3-27b", "qwen3-8b", "qwen3-30b"):
        if key in name:
            return key
    return name.replace("/", "_")


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
