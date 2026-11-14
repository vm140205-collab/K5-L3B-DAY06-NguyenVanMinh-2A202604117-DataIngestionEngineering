"""Smoke test for chunking: no API calls, checks structure and prints samples.

    python starter/scripts/smoke_chunking.py [document_id] [strategy]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document, chunk_stats  # noqa: E402
from src.config import STRATEGIES  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

doc_filter = sys.argv[1] if len(sys.argv) > 1 else ""
strategy_filter = sys.argv[2] if len(sys.argv) > 2 else ""

manifest, pages = load_corpus()
print(f"loaded {len(pages)} content pages from {len(manifest['documents'])} documents")

by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)

for name, strategy in STRATEGIES.items():
    if strategy_filter and name != strategy_filter:
        continue
    chunks = []
    for doc_id in sorted(by_doc):
        if doc_filter and doc_id != doc_filter:
            continue
        chunks.extend(chunk_document(by_doc[doc_id], strategy))
    print(f"\n=== strategy {name} ({strategy.description})")
    print(chunk_stats(chunks, name))
    for chunk in chunks[:2]:
        print(f"  [{chunk.chunk_id}] p{chunk.page} section={chunk.metadata['section']!r} "
              f"len={len(chunk.text)}")
        print("   " + chunk.text[:220].replace("\n", " | "))
    if chunks:
        last = chunks[-1]
        print(f"  last chunk p{last.page} section={last.metadata['section']!r} len={len(last.text)}")
