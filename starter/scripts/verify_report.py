"""Check every number the report claims against the artifacts that produced it.

    python starter/scripts/verify_report.py

Exits non-zero if any claim disagrees with results/. This is the review gate for
the report, not a re-run of the pipeline.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import load_questions, squash  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.config import COLLECTION, EMBEDDING_DIM, QDRANT_PATH, STRATEGIES  # noqa: E402
from src.embeddings import FastEmbedEmbedder  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402
from src.store import QdrantStore  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
REPORT = REPO_ROOT / "report" / "individual" / "2A202604117_NguyenVanMinh.md"

failures: list[str] = []
checks = 0


def check(label: str, actual, expected) -> None:
    global checks
    checks += 1
    if actual != expected:
        failures.append(f"{label}: report says {expected!r}, artifact says {actual!r}")


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


manifest = load("data/manifest.json") if False else json.loads(
    (REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))
corpus = load("corpus_stats.json")
chunks = {row["strategy"]: row for row in load("chunk_stats.json")}
metrics = load("metrics.json")
depth = load("retrieval_depth.json")
audit = load("audit_summary.json")
run_meta = load("run_meta.json")
bench = load("benchmark_validation.json")

# --- 1.1 corpus
check("documents", corpus["documents"], 14)
check("physical pages", corpus["physical_pages"], 328)
check("content pages", corpus["content_pages"], 264)
check("content chars", corpus["content_chars"], 803977)
check("years", corpus["year_values"], [2020, 2021, 2022, 2023, 2024, 2025])
check("categories", corpus["category_values"], ["cs.LG", "eess.SY", "math.OC", "stat.ML"])
counts = {}
for doc in manifest["documents"]:
    counts[doc["primary_category"]] = counts.get(doc["primary_category"], 0) + 1
check("category counts", counts, {"cs.LG": 10, "eess.SY": 2, "math.OC": 1, "stat.ML": 1})
check("every doc has CC BY 4.0",
      all(d["license_or_permission"] == "CC BY 4.0" for d in manifest["documents"]), True)
check("every doc has page_rule", all(bool(d["page_rule"]) for d in manifest["documents"]), True)

# --- 1.3 / 1.4 chunk and run facts
check("A chunks", chunks["A"]["chunks"], 1307)
check("B chunks", chunks["B"]["chunks"], 949)
check("C chunks", chunks["C"]["chunks"], 949)
check("A sections", chunks["A"]["sections"], 229)
check("B sections", chunks["B"]["sections"], 255)
check("A mean chars", chunks["A"]["mean_chars"], 729.2)
check("B mean chars", chunks["B"]["mean_chars"], 838.2)
check("A total chars", chunks["A"]["total_chars"], 953035)
check("B total chars", chunks["B"]["total_chars"], 795478)
check("A p50", chunks["A"]["p50_chars"], 794)
check("A p90", chunks["A"]["p90_chars"], 799)
check("A min", chunks["A"]["min_chars"], 160)
check("A max", chunks["A"]["max_chars"], 800)
check("B p50", chunks["B"]["p50_chars"], 923)
check("B p90", chunks["B"]["p90_chars"], 1187)
check("B min", chunks["B"]["min_chars"], 121)
check("B max", chunks["B"]["max_chars"], 1317)
check("fewer chunks pct", round((1307 - 949) / 1307 * 100), 27)
check("idempotency stable", run_meta["idempotency"]["stable"], True)
check("points after reingest", run_meta["idempotency"]["points_after_reingest"], 3205)
check("ingest seconds (3 configs)",
      round(sum(chunks[s]["ingest_seconds"] for s in "ABC")), 651)
check("total run seconds", round(run_meta["total_seconds"]), 1326)
check("embedding seconds", round(run_meta["embedding_usage"]["embed_seconds"]), 1278)
check("cost usd", run_meta["embedding_usage"]["cost_usd"], 0.0)
check("embedding model", run_meta["embedding_model"], "BAAI/bge-small-en-v1.5")

# --- 2 benchmark
check("questions", bench["questions"], 20)
check("dev", bench["dev"], 15)
check("test", bench["test"], 5)
check("filter questions", bench["filter_questions"], 3)
check("test filter questions", bench["test_filter_questions"], 1)
check("question types", bench["types"],
      {"conditional": 2, "direct": 11, "filter": 3, "paraphrase": 4})
check("benchmark ok", bench["ok"], True)
check("benchmark errors", bench["errors"], [])

# --- 3.1 metrics table
expected_metrics = {
    ("A", "dev"): ("5/15", "8/15", "10/15"),
    ("A", "test"): ("2/5", "4/5", "5/5"),
    ("A", "all"): ("7/20", "12/20", "15/20"),
    ("B", "dev"): ("7/15", "10/15", "11/15"),
    ("B", "test"): ("2/5", "3/5", "4/5"),
    ("B", "all"): ("9/20", "13/20", "15/20"),
    ("C", "dev"): ("8/15", "10/15", "11/15"),
    ("C", "test"): ("2/5", "3/5", "4/5"),
    ("C", "all"): ("10/20", "13/20", "15/20"),
}
for (strategy, split), (h1, h3, gp) in expected_metrics.items():
    row = metrics[strategy][split]
    check(f"{strategy}/{split} hit@1", row["hit_at_1"], h1)
    check(f"{strategy}/{split} hit@3", row["hit_at_3"], h3)
    check(f"{strategy}/{split} gold page", row["gold_page_in_top3"], gp)
check("A filter compliant all", metrics["A"]["all"]["filter_compliant"], "2/3")
check("B filter compliant all", metrics["B"]["all"]["filter_compliant"], "2/3")
check("C filter compliant all", metrics["C"]["all"]["filter_compliant"], "3/3")
check("A filter dev", metrics["A"]["dev"]["filter_compliant"], "1/2")
check("C filter dev", metrics["C"]["dev"]["filter_compliant"], "2/2")

# --- 3.2 audit
check("coherence A", audit["coherence"]["A"]["mean"], 1.0)
check("coherence B", audit["coherence"]["B"]["mean"], 1.5)
check("coherence A distribution", audit["coherence"]["A"]["distribution"], {"1": 10})
check("coherence B distribution", audit["coherence"]["B"]["distribution"], {"1": 5, "2": 5})
for strategy in "ABC":
    check(f"metadata {strategy}", audit["metadata"][strategy]["records_correct"], 15)
    check(f"metadata {strategy} checked", audit["metadata"][strategy]["records_checked"], 15)
    check(f"grounding {strategy} mean", audit["grounding"][strategy]["mean"], 1.8)
    check(f"grounding {strategy} dist", audit["grounding"][strategy]["distribution"],
          {"1": 1, "2": 4})

# --- 3.3 depth table
expected_depth = {
    "A": (7, 12, 14, 14, 17),
    "B": (9, 13, 14, 15, 17),
    "C": (10, 13, 14, 15, 17),
}
for strategy, (t1, t3, t10, t20, gp20) in expected_depth.items():
    s = depth["summary"][strategy]
    check(f"{strategy} depth top1", s["evidence_in_top1"], t1)
    check(f"{strategy} depth top3", s["evidence_in_top3"], t3)
    check(f"{strategy} depth top10", s["evidence_in_top10"], t10)
    check(f"{strategy} depth top20", s["evidence_in_top20"], t20)
    check(f"{strategy} gold page top20", s["gold_page_in_top20"], gp20)
check("A never found count", len(depth["summary"]["A"]["evidence_never_found"]), 6)
check("B never found count", len(depth["summary"]["B"]["evidence_never_found"]), 5)

rows = {(r["strategy"], r["question_id"]): r for r in depth["rows"]}
check("q20 rank A", rows[("A", "q20")]["evidence_rank_in_20"], 3)
check("q20 rank B", rows[("B", "q20")]["evidence_rank_in_20"], 4)
check("q10 rank A", rows[("A", "q10")]["evidence_rank_in_20"], None)
check("q10 rank B", rows[("B", "q10")]["evidence_rank_in_20"], 1)
check("q14 rank A", rows[("A", "q14")]["evidence_rank_in_20"], None)
check("q14 rank B", rows[("B", "q14")]["evidence_rank_in_20"], 2)
check("q06 rank A", rows[("A", "q06")]["evidence_rank_in_20"], 7)
check("q06 rank B", rows[("B", "q06")]["evidence_rank_in_20"], None)

# --- 4.1 title-only chunks
_, pages = load_corpus()
by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)
title_only = []
for doc_id in sorted(by_doc):
    first = next((c for c in chunk_document(by_doc[doc_id], STRATEGIES["B"]) if c.page == 1), None)
    if first is not None and not re.search(r"[.!?](\s|$)", first.text):
        title_only.append(doc_id)
check("title-only page-1 chunks",
      title_only, ["2104.10383v1", "2303.13117v1", "2305.03360v1", "2308.14897v1", "2505.22866v1"])

# --- 4.3 q01 cosine and q14 top-1 in B
questions = {q["id"]: q for q in load_questions()}
embedder = FastEmbedEmbedder()
store = QdrantStore(QDRANT_PATH, COLLECTION, EMBEDDING_DIM)
# A fresh clone has no vector store; the four checks below need it. Everything
# else is verified from committed JSON/CSV, so this degrades instead of failing.
try:
    has_store = store.client.collection_exists(COLLECTION) and store.count("A") > 0
except Exception:  # noqa: BLE001 - a fresh clone has no collection at all
    has_store = False
if not has_store:
    print("NOTE: no vector store found; skipping the 4 checks that re-run a search."
          " Run `cd starter && python -m src.run` first to enable them.")

if has_store:
    a_chunks = []
    for doc_id in sorted(by_doc):
        a_chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES["A"]))
    q01 = questions["q01"]
    quote = squash(q01["evidence_quote"])
    holder = [c for c in a_chunks if c.document_id == q01["gold_source_id"]
              and c.page in q01["gold_pages"] and quote in squash(c.text)]
    check("q01 evidence intact in one A chunk", len(holder), 1)
    if holder:
        import numpy as np
        vec = np.array(embedder.embed_documents([holder[0].text])[0])
        qv = np.array(embedder.embed_query(q01["question"]))
        cos = float(vec @ qv / (np.linalg.norm(vec) * np.linalg.norm(qv)))
        check("q01 evidence cosine (3dp)", round(cos, 3), 0.687)
    top1 = store.search("A", embedder.embed_query(q01["question"]), 1)[0]
    check("q01 top1 score (3dp)", round(top1["score"], 3), 0.791)
    check("q01 top1 document", top1["document_id"], "2506.17155v2")

    # Strategy B applies no filter, so this must be an unfiltered search.
    b_q14 = store.search("B", embedder.embed_query(questions["q14"]["question"]), 3)
    check("q14 B rank1 year", b_q14[0]["year"], 2025)
    check("q14 B rank1 doc/page", (b_q14[0]["document_id"], b_q14[0]["page"]),
          ("2505.22866v1", 32))
    check("q14 B rank2 is gold", (b_q14[1]["document_id"], b_q14[1]["page"]),
          ("2410.14069v1", 1))
store.close()

# --- report file itself: style and required sections
text = REPORT.read_text(encoding="utf-8")
for section in ("## 1. Dữ liệu và pipeline", "## 2. Cấu hình và benchmark", "## 3. Kết quả",
                "## 4. Phân tích và kết luận", "## 5. Công cụ AI", "## 6. Bảng đối chiếu"):
    check(f"report has section {section!r}", section in text, True)
check("report has no curly quotes",
      any(ch in text for ch in "\u201c\u201d\u2018\u2019"), False)
check("report em dashes (title only)", text.count("\u2014"), 1)

print(f"{checks - len(failures)}/{checks} checks passed")
for failure in failures:
    print(f"  FAIL {failure}")
raise SystemExit(1 if failures else 0)
