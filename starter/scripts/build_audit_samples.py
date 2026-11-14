"""Build the manual-audit samples and the objective part of the metadata audit.

Positions for the coherence sample are fixed before any score is looked at:
the first page of the 10 first documents by source_id.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import load_questions, squash  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.config import RESULTS_DIR, STRATEGIES  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

_, pages = load_corpus()
manifest = json.loads((Path(__file__).resolve().parents[2] / "data" / "manifest.json")
                      .read_text(encoding="utf-8"))
doc_meta = {d["source_id"]: d for d in manifest["documents"]}
page_text = {(p.document_id, p.page): p.text for p in pages}

by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)

chunks_by_strategy: dict[str, list] = {}
for name in ("A", "B"):
    chunks = []
    for doc_id in sorted(by_doc):
        chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES[name]))
    chunks_by_strategy[name] = chunks

# ---------------------------------------------------------------- coherence
positions = [(doc_id, 1) for doc_id in sorted(by_doc)[:10]]
coh_rows = []
for name in ("A", "B"):
    for doc_id, page_no in positions:
        chunk = next((c for c in chunks_by_strategy[name]
                      if c.document_id == doc_id and c.page == page_no), None)
        coh_rows.append({
            "strategy": name,
            "document_id": doc_id,
            "page": page_no,
            "section": chunk.metadata["section"] if chunk else "",
            "char_len": len(chunk.text) if chunk else 0,
            "chunk_text": chunk.text if chunk else "",
            "coherence_0_1_2": "",
            "reason": "",
        })

with (RESULTS_DIR / "audit_coherence.csv").open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(coh_rows[0]))
    writer.writeheader()
    writer.writerows(coh_rows)

print(f"coherence sample: {len(coh_rows)} rows -> results/audit_coherence.csv")
for row in coh_rows:
    print(f"\n--- {row['strategy']} {row['document_id']} p{row['page']} "
          f"section={row['section']!r} len={row['char_len']}")
    print(row["chunk_text"][:900])

# ---------------------------------------------------------------- grounding
questions = load_questions()
test_qs = [q for q in questions if q["split"] == "test"]
gr_rows = []
for name in ("A", "B", "C"):
    traces = {json.loads(line)["question_id"]: json.loads(line)
              for line in (RESULTS_DIR / f"traces_{name}.jsonl").open(encoding="utf-8")}
    for q in test_qs:
        trace = traces[q["id"]]
        quote = squash(q["evidence_quote"])
        evidence_ranks = [h["rank"] for h in trace["hits"] if quote in squash(h["text"])]
        gr_rows.append({
            "strategy": name,
            "question_id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "evidence_quote": q["evidence_quote"],
            "evidence_present_at_rank": "|".join(str(r) for r in evidence_ranks) or "none",
            "top1": f"{trace['hits'][0]['document_id']} p{trace['hits'][0]['page']}",
            "top1_excerpt": trace["hits"][0]["text"][:300],
            "top2_excerpt": trace["hits"][1]["text"][:300],
            "top3_excerpt": trace["hits"][2]["text"][:300],
            "grounding_0_1_2": "",
            "reason": "",
        })

with (RESULTS_DIR / "audit_grounding.csv").open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(gr_rows[0]))
    writer.writeheader()
    writer.writerows(gr_rows)
print(f"\ngrounding sample: {len(gr_rows)} rows -> results/audit_grounding.csv")

# ---------------------------------------------------------------- metadata
md_rows = []
for name in ("A", "B", "C"):
    traces = [json.loads(line) for line in (RESULTS_DIR / f"traces_{name}.jsonl").open(encoding="utf-8")]
    for trace in traces:
        if trace["split"] != "test":
            continue
        for hit in trace["hits"]:
            key = (hit["document_id"], hit["page"])
            text_on_page = page_text.get(key, "")
            meta = doc_meta.get(hit["document_id"], {})
            text_ok = squash(hit["text"])[:200] in squash(text_on_page)
            md_rows.append({
                "strategy": name,
                "question_id": trace["question_id"],
                "rank": hit["rank"],
                "document_id": hit["document_id"],
                "page": hit["page"],
                "source_matches_manifest": hit["source"] == meta.get("source_path_or_url"),
                "year_matches_manifest": hit["year"] == meta.get("year"),
                "category_matches_manifest": hit["primary_category"] == meta.get("primary_category"),
                "text_found_on_claimed_page": text_ok,
                "section": hit["section"],
                "filter_required": bool(trace["filter"]),
                "filter_satisfied": all(hit.get(k) == v for k, v in trace["filter"].items())
                if trace["filter"] else "",
            })

with (RESULTS_DIR / "audit_metadata.csv").open("w", encoding="utf-8", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(md_rows[0]))
    writer.writeheader()
    writer.writerows(md_rows)

total = len(md_rows)
ok = sum(1 for r in md_rows if r["text_found_on_claimed_page"] and r["source_matches_manifest"]
         and r["year_matches_manifest"] and r["category_matches_manifest"])
print(f"metadata audit: {ok}/{total} records have correct source/page/year/category "
      f"and text found on the claimed page")
