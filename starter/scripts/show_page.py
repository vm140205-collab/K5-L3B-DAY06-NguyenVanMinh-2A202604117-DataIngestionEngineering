"""Dump the normalized text of one page (for eyeballing structure)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_corpus  # noqa: E402

doc = sys.argv[1]
page_no = int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
_, pages = load_corpus()
for page in pages:
    if page.document_id == doc and page.page == page_no:
        print(page.text[:limit])
        break
