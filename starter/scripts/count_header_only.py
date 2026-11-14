"""Count B chunks that contain no sentence at all (title/author/heading blocks)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document  # noqa: E402
from src.config import STRATEGIES  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

_, pages = load_corpus()
by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)

chunks = []
for doc_id in sorted(by_doc):
    chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES["B"]))

no_sentence = [c for c in chunks if not re.search(r"[.!?](\s|$)", c.text)]
print(f"B: {len(no_sentence)}/{len(chunks)} chunks contain no sentence at all")
for c in no_sentence:
    print(f"  {c.document_id} p{c.page} len={len(c.text)} :: {c.text[:110]!r}")

for name in ("A",):
    other = []
    for doc_id in sorted(by_doc):
        other.extend(chunk_document(by_doc[doc_id], STRATEGIES[name]))
    n = sum(1 for c in other if not re.search(r"[.!?](\s|$)", c.text))
    print(f"{name}: {n}/{len(other)} chunks contain no sentence at all")
