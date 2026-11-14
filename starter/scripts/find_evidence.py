"""Find verbatim sentences in the corpus, for authoring evidence quotes.

    python starter/scripts/find_evidence.py "proximal policy" [max_hits]

Prints doc, page and the sentence as a single line so it can be pasted into
benchmark/questions.jsonl unchanged (evidence matching is whitespace-insensitive).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion import load_corpus  # noqa: E402

pattern = re.compile(sys.argv[1], re.I)
max_hits = int(sys.argv[2]) if len(sys.argv) > 2 else 12
_, pages = load_corpus()

hits = 0
for page in pages:
    flat = re.sub(r"\s+", " ", page.text)
    for sentence in re.split(r"(?<=[.!?])\s+", flat):
        if pattern.search(sentence):
            print(f"{page.document_id} p{page.page}: {sentence.strip()}")
            hits += 1
            if hits >= max_hits:
                raise SystemExit(0)
