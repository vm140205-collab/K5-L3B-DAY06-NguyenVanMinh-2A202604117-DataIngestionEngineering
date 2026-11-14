"""Audit the content-page rule: show which pages were dropped and why.

Prints, per document, every non-content page with a sample line, so the page
accounting in data/manifest.json can be checked by eye instead of trusted.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
pages = [json.loads(line) for line in (REPO_ROOT / "data" / "pages.jsonl").open(encoding="utf-8")]
manifest = json.loads((REPO_ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))

verbose = "--verbose" in sys.argv
total_content = 0
for doc in manifest["documents"]:
    did = doc["source_id"]
    rows = [p for p in pages if p["document_id"] == did]
    dropped = [p for p in rows if p["page_kind"] != "content"]
    total_content += doc["content_page_count"]
    print(f"\n=== {did}  physical={doc['physical_pages']} content={doc['content_page_count']} "
          f"dropped={len(dropped)}")
    for p in dropped:
        sample = ""
        for line in p["text"].splitlines():
            if line.strip():
                sample = line.strip()[:100]
                break
        print(f"  p{p['page']:3d} {p['page_kind']:12s} chars={p['body_chars']:5d} | {sample}")

print(f"\nTOTAL content pages = {total_content} (manifest says {manifest['totals']['content_pages']})")
if verbose:
    for doc in manifest["documents"]:
        print(f"{doc['source_id']}: dropped headers -> {doc['dropped_running_headers']}")
