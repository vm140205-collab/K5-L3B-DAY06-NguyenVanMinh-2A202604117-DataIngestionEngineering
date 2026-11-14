"""Print the first page of every document (title + abstract) for question authoring."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_corpus  # noqa: E402

limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
manifest, pages = load_corpus()
meta = {d["source_id"]: d for d in manifest["documents"]}
for doc_id in sorted(meta):
    first = next((p for p in pages if p.document_id == doc_id and p.page == 1), None)
    d = meta[doc_id]
    print(f"\n{'=' * 90}\n{doc_id} | {d['year']} | {d['primary_category']} | {d['title']}")
    if first:
        print(first.text[:limit])
