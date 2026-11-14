"""Day 06 pipeline: corpus -> chunks -> embeddings -> Qdrant -> benchmark traces.

    python -m src.run                 # full run (ingest + benchmark)
    python -m src.run --recreate      # drop the collection first
    python -m src.run --skip-ingest   # reuse stored vectors, only re-run the benchmark

Run it from the `starter/` directory. Requires MISTRAL_API_KEY in the environment.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone

from .benchmark import load_questions, validate
from .chunking import chunk_document, chunk_stats
from .config import (COLLECTION, EMBEDDING_DIM, EMBEDDING_MODEL, QDRANT_PATH,
                     RESULTS_DIR, STRATEGIES, TOP_K)
from .embeddings import FastEmbedEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import corpus_stats, load_corpus
from .store import QdrantStore

# Local embedding model: no metered cost. Kept as an explicit constant so the
# report can state the cost line as a computed value rather than a guess.
USD_PER_1M_TOKENS = 0.0


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_chunks(pages, strategy_name: str):
    strategy = STRATEGIES[strategy_name]
    by_doc: dict[str, list] = {}
    for page in pages:
        by_doc.setdefault(page.document_id, []).append(page)
    chunks = []
    for doc_id in sorted(by_doc):
        chunks.extend(chunk_document(by_doc[doc_id], strategy))
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true", help="drop the Qdrant collection first")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse stored vectors")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    run_meta: dict = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "vector_store": "qdrant-client embedded (local persistent)",
        "collection": COLLECTION,
        "top_k": TOP_K,
        "strategies": {k: v.__dict__ for k, v in STRATEGIES.items()},
    }

    manifest, pages = load_corpus()
    stats = corpus_stats(manifest, pages)
    if stats["content_pages"] < 30:
        raise SystemExit(f"corpus has {stats['content_pages']} content pages, need >= 30")
    print(f"corpus: {stats['documents']} docs, {stats['content_pages']} content pages, "
          f"{stats['content_chars']} chars")
    _write_json(RESULTS_DIR / "corpus_stats.json", stats)

    questions = load_questions()
    check = validate(questions, pages, manifest)
    _write_json(RESULTS_DIR / "benchmark_validation.json", check)
    print(f"benchmark: {check['questions']} questions "
          f"(dev {check['dev']} / test {check['test']}, filter {check['filter_questions']})")
    if not check["ok"]:
        for err in check["errors"]:
            print(f"  BENCHMARK ERROR: {err}")
        raise SystemExit("benchmark validation failed; fix the question set first")

    store = QdrantStore(QDRANT_PATH, COLLECTION, EMBEDDING_DIM)
    store.ensure_collection(recreate=args.recreate)

    chunk_counts: dict[str, int] = {}
    stats_rows = []
    if not args.skip_ingest:
        embedder = FastEmbedEmbedder()
        for name in STRATEGIES:
            t0 = time.time()
            chunks = build_chunks(pages, name)
            vectors = embedder.embed_documents([c.text for c in chunks])
            written = store.upsert(name, chunks, vectors)
            elapsed = time.time() - t0
            chunk_counts[name] = written["upserted"]
            row = chunk_stats(chunks, name)
            row.update({"ingest_seconds": round(elapsed, 1),
                        "deleted_stale": written["deleted_stale"]})
            stats_rows.append(row)
            print(f"strategy {name}: {written['upserted']} chunks, "
                  f"{row['mean_chars']} mean chars, {elapsed:.1f}s")

        # Idempotency check: the same chunks must not double on a second write.
        before = store.count()
        for name in STRATEGIES:
            chunks = build_chunks(pages, name)
            vectors = embedder.embed_documents([c.text for c in chunks])
            store.upsert(name, chunks, vectors)
        after = store.count()
        run_meta["idempotency"] = {"points_before_reingest": before,
                                   "points_after_reingest": after,
                                   "stable": before == after}
        run_meta["embedding_usage"] = embedder.usage()
        _write_json(RESULTS_DIR / "chunk_stats.json", stats_rows)
        print(f"idempotency: {before} -> {after} points ({'stable' if before == after else 'DRIFT'})")

    traces = []
    query_embedder = FastEmbedEmbedder()
    for name, strategy in STRATEGIES.items():
        t0 = time.time()
        strategy_traces = []
        for question in questions:
            filters = question.get("filter") if strategy.use_metadata_filter else None
            vector = query_embedder.embed_query(question["question"])
            hits = store.search(name, vector, TOP_K, filters)
            trace = evaluate_query(question, hits)
            trace["strategy"] = name
            trace["applied_filter"] = filters or {}
            strategy_traces.append(trace)
        traces.extend(strategy_traces)
        with (RESULTS_DIR / f"traces_{name}.jsonl").open("w", encoding="utf-8") as fh:
            for trace in strategy_traces:
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
        print(f"strategy {name}: {sum(t['hit_at_3'] for t in strategy_traces)}/"
              f"{len(strategy_traces)} evidence hits in top-3 ({time.time() - t0:.1f}s)")

    metrics = summarize(traces)
    _write_json(RESULTS_DIR / "metrics.json", metrics)
    run_meta["stored_points"] = {name: store.count(name) for name in STRATEGIES}
    run_meta["query_usage"] = query_embedder.usage()
    run_meta["total_seconds"] = round(time.time() - started, 1)
    _write_json(RESULTS_DIR / "run_meta.json", run_meta)
    store.close()

    for name in STRATEGIES:
        row = metrics[name]["all"]
        print(f"{name}: Hit@1 {row['hit_at_1']} ({row['hit_at_1_rate']:.1%})  "
              f"Evidence Hit@3 {row['hit_at_3']} ({row['hit_at_3_rate']:.1%})")
    print(f"done in {run_meta['total_seconds']}s -> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
