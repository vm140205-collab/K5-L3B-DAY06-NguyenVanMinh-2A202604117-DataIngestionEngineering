"""Verify the specific claims made in the report against the artifacts."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document  # noqa: E402
from src.config import RESULTS_DIR, STRATEGIES  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

_, pages = load_corpus()
by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)

chunks = []
for doc_id in sorted(by_doc):
    chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES["B"]))

# A "header-only" chunk: short and ends on a heading word rather than a sentence.
header_only = [
    c for c in chunks
    if len(c.text) < 400 and not re.search(r"[.!?]\s*$", c.text.strip())
]
print(f"B chunks that end without sentence punctuation and are < 400 chars: {len(header_only)}")
for c in header_only:
    print(f"  {c.document_id} p{c.page} len={len(c.text)} tail={c.text[-60:]!r}")

print("\nB filter-question hits:")
for line in (RESULTS_DIR / "traces_B.jsonl").open(encoding="utf-8"):
    trace = json.loads(line)
    if not trace["filter"]:
        continue
    ok = all(all(h.get(k) == v for k, v in trace["filter"].items()) for h in trace["hits"])
    print(f"  {trace['question_id']} filter={trace['filter']} compliant={ok} hit@3={trace['hit_at_3']}")
    for h in trace["hits"]:
        print(f"    rank{h['rank']} {h['document_id']} p{h['page']} year={h['year']} "
              f"cat={h['primary_category']}")
