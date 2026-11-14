"""Probe: compare flat text vs layout blocks for one PDF page."""
import sys
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[2]
pdf = REPO_ROOT / "data" / "corpus" / (sys.argv[1] if len(sys.argv) > 1 else "2004.01625v2.pdf")
page_no = int(sys.argv[2]) if len(sys.argv) > 2 else 2

doc = pymupdf.open(pdf)
page = doc[page_no - 1]
print("=== flat text (first 300 chars) ===")
print(page.get_text("text")[:300])
print(f"\n=== blocks: {len(page.get_text('blocks'))} ===")
for i, block in enumerate(page.get_text("blocks")):
    x0, y0, x1, y1, text, block_no, block_type = block
    text = " ".join(text.split())
    print(f"[{i}] type={block_type} x0={x0:.0f} y0={y0:.0f} len={len(text)} :: {text[:120]}")
doc.close()
