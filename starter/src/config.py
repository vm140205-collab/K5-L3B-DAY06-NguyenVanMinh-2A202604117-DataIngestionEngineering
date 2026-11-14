"""Single source of truth for paths, model identity and the three strategies."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
# starter/.env holds MISTRAL_API_KEY locally; it is git-ignored.
load_dotenv(REPO_ROOT / "starter" / ".env")
DATA_DIR = REPO_ROOT / "data"
PAGES_PATH = DATA_DIR / "pages.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"
BENCHMARK_PATH = REPO_ROOT / "benchmark" / "questions.jsonl"
RESULTS_DIR = REPO_ROOT / "results"
QDRANT_PATH = REPO_ROOT / "qdrant_data"
COLLECTION = "day6_chunks"

# One real embedding model for every strategy (GUIDE.md section 3).
# Local ONNX model: no API key, no per-token cost, reproducible offline.
# (The Mistral API key available at build time expired 2026-11-14; see the report.)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
EMBEDDING_BATCH = 64

TOP_K = 3


@dataclass(frozen=True)
class Strategy:
    """A data-preparation configuration. A vs B changes chunking; B vs C changes filtering."""

    name: str
    chunker: str            # "fixed" | "paragraph"
    size: int
    overlap: int
    use_metadata_filter: bool
    description: str


STRATEGIES: dict[str, Strategy] = {
    "A": Strategy(
        name="A",
        chunker="fixed",
        size=800,
        overlap=150,
        use_metadata_filter=False,
        description="fixed 800-char window, 150-char overlap, page-scoped, no filter",
    ),
    "B": Strategy(
        name="B",
        chunker="paragraph",
        size=1200,
        overlap=0,
        use_metadata_filter=False,
        description="paragraph/section-aware packing up to 1200 chars, page-scoped, no filter",
    ),
    "C": Strategy(
        name="C",
        chunker="paragraph",
        size=1200,
        overlap=0,
        use_metadata_filter=True,
        description="same chunks as B, metadata filter applied when the question supplies one",
    ),
}
