"""Diagnostic: print per-page accounting for one or more documents."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
pages = [json.loads(line) for line in (REPO_ROOT / "data" / "pages.jsonl").open(encoding="utf-8")]

targets = sys.argv[1:] or sorted({p["document_id"] for p in pages})
for doc in targets:
    print(f"===== {doc}")
    for row in pages:
        if row["document_id"] != doc:
            continue
        head = row["text"][:90].replace("\n", " | ")
        print(f"p{row['page']:3d} body={row['body_chars']:5d} "
              f"content={str(row['is_content_page']):5s} :: {head}")
