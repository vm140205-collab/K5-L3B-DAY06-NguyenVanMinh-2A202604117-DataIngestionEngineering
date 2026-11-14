"""Diagnostic: how deep must we search before the gold evidence chunk appears?

Hit@3 is the headline metric; this shows whether a miss is "rank 4" or "not even
in the top 20", which separates ranking problems from representation problems.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import load_questions, squash  # noqa: E402
from src.config import COLLECTION, EMBEDDING_DIM, QDRANT_PATH, RESULTS_DIR  # noqa: E402
from src.embeddings import FastEmbedEmbedder  # noqa: E402
from src.store import QdrantStore  # noqa: E402

DEPTH = 20
questions = load_questions()
embedder = FastEmbedEmbedder()
store = QdrantStore(QDRANT_PATH, COLLECTION, EMBEDDING_DIM)

rows = []
for strategy in ("A", "B", "C"):
    for q in questions:
        filters = q.get("filter") if strategy == "C" else None
        vector = embedder.embed_query(q["question"])
        hits = store.search(strategy, vector, DEPTH, filters)
        rank = next((h["rank"] for h in hits
                     if h["document_id"] == q["gold_source_id"]
                     and h["page"] in q["gold_pages"]
                     and squash(q["evidence_quote"]) in squash(h["text"])), None)
        page_rank = next((h["rank"] for h in hits
                          if h["document_id"] == q["gold_source_id"]
                          and h["page"] in q["gold_pages"]), None)
        rows.append({"strategy": strategy, "question_id": q["id"], "split": q["split"],
                     "evidence_rank_in_20": rank, "gold_page_rank_in_20": page_rank})

summary = {}
for strategy in ("A", "B", "C"):
    subset = [r for r in rows if r["strategy"] == strategy]
    summary[strategy] = {
        "questions": len(subset),
        "evidence_in_top1": sum(1 for r in subset if r["evidence_rank_in_20"] == 1),
        "evidence_in_top3": sum(1 for r in subset if (r["evidence_rank_in_20"] or 99) <= 3),
        "evidence_in_top10": sum(1 for r in subset if (r["evidence_rank_in_20"] or 99) <= 10),
        "evidence_in_top20": sum(1 for r in subset if r["evidence_rank_in_20"] is not None),
        "gold_page_in_top20": sum(1 for r in subset if r["gold_page_rank_in_20"] is not None),
        "evidence_never_found": [r["question_id"] for r in subset
                                 if r["evidence_rank_in_20"] is None],
    }
    print(f"{strategy}: {summary[strategy]}")

(RESULTS_DIR / "retrieval_depth.json").write_text(
    json.dumps({"depth": DEPTH, "summary": summary, "rows": rows}, indent=2), encoding="utf-8")
store.close()
