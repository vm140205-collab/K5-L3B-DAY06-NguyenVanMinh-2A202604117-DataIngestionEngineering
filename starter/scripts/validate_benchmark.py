"""Validate benchmark/questions.jsonl against the corpus text before any run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import load_questions, validate  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

manifest, pages = load_corpus()
questions = load_questions()
report = validate(questions, pages, manifest)
for key, value in report.items():
    if key != "errors":
        print(f"{key}: {value}")
if report["errors"]:
    print("\nERRORS:")
    for err in report["errors"]:
        print(f"  - {err}")
raise SystemExit(0 if report["ok"] else 1)
