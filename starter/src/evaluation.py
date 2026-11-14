"""Retrieval metrics and per-query traces.

Hit definition (GUIDE.md section 4): a retrieved chunk is a hit when it comes from
the gold document, sits on a gold page, and contains the recorded evidence quote.
A quote split across two chunks is therefore a miss; that case is counted
separately as `evidence_split` so the failure mode stays visible.
"""
from __future__ import annotations

from typing import Any

from .benchmark import squash


def _is_hit(question: dict, hit: dict) -> bool:
    if hit.get("document_id") != question["gold_source_id"]:
        return False
    if hit.get("page") not in question["gold_pages"]:
        return False
    return squash(question["evidence_quote"]) in squash(hit.get("text", ""))


def _on_gold_page(question: dict, hit: dict) -> bool:
    return (hit.get("document_id") == question["gold_source_id"]
            and hit.get("page") in question["gold_pages"])


def evaluate_query(question: dict, hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one question's top-k list and keep the full trace."""
    hit_flags = [_is_hit(question, h) for h in hits]
    page_flags = [_on_gold_page(question, h) for h in hits]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type", "untyped"),
        "filter": question.get("filter") or {},
        "hit_at_1": bool(hit_flags[:1] and hit_flags[0]),
        "hit_at_3": any(hit_flags[:3]),
        "gold_page_in_top3": any(page_flags[:3]),
        "evidence_split": any(page_flags[:3]) and not any(hit_flags[:3]),
        "first_hit_rank": next((i + 1 for i, f in enumerate(hit_flags) if f), None),
        "hits": hits,
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 / Evidence Hit@3 per strategy and split, with counts and rates."""
    out: dict[str, Any] = {}
    for strategy in sorted({t["strategy"] for t in traces}):
        out[strategy] = {}
        for split in ("dev", "test", "all"):
            subset = [t for t in traces
                      if t["strategy"] == strategy and (split == "all" or t["split"] == split)]
            if not subset:
                continue
            n = len(subset)
            h1 = sum(1 for t in subset if t["hit_at_1"])
            h3 = sum(1 for t in subset if t["hit_at_3"])
            gp = sum(1 for t in subset if t["gold_page_in_top3"])
            split_evidence = sum(1 for t in subset if t["evidence_split"])
            filtered = [t for t in subset if t["filter"]]
            compliant = 0
            for t in filtered:
                ok = all(all(h.get(k) == v for k, v in t["filter"].items()) for h in t["hits"])
                compliant += 1 if ok else 0
            out[strategy][split] = {
                "questions": n,
                "hit_at_1": f"{h1}/{n}",
                "hit_at_1_rate": round(h1 / n, 4),
                "hit_at_3": f"{h3}/{n}",
                "hit_at_3_rate": round(h3 / n, 4),
                "gold_page_in_top3": f"{gp}/{n}",
                "evidence_split_misses": split_evidence,
                "filter_questions": len(filtered),
                "filter_compliant": f"{compliant}/{len(filtered)}" if filtered else "n/a",
            }
    return out
