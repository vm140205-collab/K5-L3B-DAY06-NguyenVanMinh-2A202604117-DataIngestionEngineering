"""Count page-1 first chunks in B that hold only the title/author block."""
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

title_only = 0
for doc_id in sorted(by_doc):
    chunks = chunk_document(by_doc[doc_id], STRATEGIES["B"])
    first = next((c for c in chunks if c.page == 1), None)
    if first is None:
        continue
    has_sentence = bool(re.search(r"[.!?](\s|$)", first.text))
    title_only += 0 if has_sentence else 1
    print(f"{doc_id}: len={len(first.text):4d} sentence={has_sentence} "
          f"tail={first.text[-50:]!r}")
print(f"\nB: {title_only} documents where the page-1 chunk contains no sentence at all")
