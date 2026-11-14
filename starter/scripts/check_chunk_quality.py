"""Report chunk-quality problems: tiny chunks, section coverage, page coverage."""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.chunking import chunk_document  # noqa: E402
from src.config import STRATEGIES  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402

manifest, pages = load_corpus()
by_doc: dict[str, list] = defaultdict(list)
for page in pages:
    by_doc[page.document_id].append(page)

for name in ("A", "B"):
    chunks = []
    for doc_id in sorted(by_doc):
        chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES[name]))
    tiny = [c for c in chunks if len(c.text) < 120]
    print(f"\n=== {name}: {len(chunks)} chunks, {len(tiny)} under 120 chars")
    for c in tiny[:10]:
        print(f"  {c.document_id} p{c.page} len={len(c.text)} {c.text[:60]!r}")

    per_doc_sections = Counter()
    for c in chunks:
        if c.metadata["section"]:
            per_doc_sections[c.document_id] += 1
    no_section = sum(1 for c in chunks if not c.metadata["section"])
    print(f"  chunks without section label: {no_section}/{len(chunks)}")
    print(f"  distinct sections overall: {len({c.metadata['section'] for c in chunks if c.metadata['section']})}")

    covered = {(c.document_id, c.page) for c in chunks}
    missing = [(p.document_id, p.page) for p in pages if (p.document_id, p.page) not in covered]
    print(f"  content pages with no chunk: {len(missing)} {missing[:5]}")
    lens = [len(c.text) for c in chunks]
    print(f"  min={min(lens)} p50={sorted(lens)[len(lens)//2]} max={max(lens)}")
