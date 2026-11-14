"""Load and validate the question set before any retrieval run."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import BENCHMARK_PATH
from .models import DocumentPage

REQUIRED = ("id", "split", "question", "expected_answer", "gold_source_id",
            "gold_pages", "evidence_quote")
MIN_TOTAL, MIN_DEV, MIN_TEST, MIN_FILTER, MIN_TEST_FILTER = 20, 15, 5, 3, 1


def squash(text: str) -> str:
    """Whitespace-insensitive form used for evidence matching."""
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions(path: Path = BENCHMARK_PATH) -> list[dict]:
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def validate(questions: list[dict], pages: list[DocumentPage],
             manifest: dict) -> dict:
    """Check the question set against the corpus text; no question is trusted blindly."""
    page_text = {(p.document_id, p.page): squash(p.text) for p in pages}
    content_pages = {(p.document_id, p.page) for p in pages}
    years = {d["year"] for d in manifest["documents"]}
    categories = {d["primary_category"] for d in manifest["documents"]}
    errors: list[str] = []

    ids = [q.get("id") for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")

    for q in questions:
        qid = q.get("id", "?")
        missing = [f for f in REQUIRED if not q.get(f) and q.get(f) != 0]
        if missing:
            errors.append(f"{qid}: missing fields {missing}")
            continue
        if q["split"] not in ("dev", "test"):
            errors.append(f"{qid}: split must be dev|test")
        if not isinstance(q["gold_pages"], list) or not q["gold_pages"]:
            errors.append(f"{qid}: gold_pages must be a non-empty list")
            continue
        for page in q["gold_pages"]:
            if (q["gold_source_id"], page) not in content_pages:
                errors.append(f"{qid}: gold page {q['gold_source_id']} p{page} is not a content page")
        quote = squash(q["evidence_quote"])
        if not quote:
            errors.append(f"{qid}: empty evidence_quote")
        elif not any(quote in page_text.get((q["gold_source_id"], p), "")
                     for p in q["gold_pages"]):
            errors.append(f"{qid}: evidence_quote not found verbatim on gold page(s)")
        filt = q.get("filter") or {}
        for key, value in filt.items():
            if key == "year" and value not in years:
                errors.append(f"{qid}: filter year {value} not in corpus")
            if key == "primary_category" and value not in categories:
                errors.append(f"{qid}: filter category {value} not in corpus")

    splits = {s: sum(1 for q in questions if q.get("split") == s) for s in ("dev", "test")}
    filters = [q for q in questions if q.get("filter")]
    test_filters = [q for q in filters if q.get("split") == "test"]
    if len(questions) < MIN_TOTAL:
        errors.append(f"only {len(questions)} questions, need >= {MIN_TOTAL}")
    if splits["dev"] < MIN_DEV:
        errors.append(f"only {splits['dev']} dev questions, need >= {MIN_DEV}")
    if splits["test"] < MIN_TEST:
        errors.append(f"only {splits['test']} test questions, need >= {MIN_TEST}")
    if len(filters) < MIN_FILTER:
        errors.append(f"only {len(filters)} filter questions, need >= {MIN_FILTER}")
    if len(test_filters) < MIN_TEST_FILTER:
        errors.append(f"only {len(test_filters)} test filter questions, need >= {MIN_TEST_FILTER}")

    return {
        "questions": len(questions),
        "dev": splits["dev"],
        "test": splits["test"],
        "filter_questions": len(filters),
        "test_filter_questions": len(test_filters),
        "types": _count(q.get("type", "untyped") for q in questions),
        "errors": errors,
        "ok": not errors,
    }


def _count(values) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
