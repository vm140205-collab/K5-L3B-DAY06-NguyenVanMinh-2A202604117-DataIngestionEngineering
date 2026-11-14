"""Print the detected section labels and the first lines of a document's pages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import heading_of, load_corpus, split_paragraphs  # noqa: E402

doc = sys.argv[1] if len(sys.argv) > 1 else "2004.01625v2"
_, pages = load_corpus()
for page in pages:
    if page.document_id != doc:
        continue
    heads = [h for h in (heading_of(p) for _, p in split_paragraphs(page.text)) if h]
    print(f"p{page.page}: {heads}")
    if "--paras" in sys.argv:
        for _, para in split_paragraphs(page.text)[:4]:
            print(f"    {para.splitlines()[0][:90]!r}")
