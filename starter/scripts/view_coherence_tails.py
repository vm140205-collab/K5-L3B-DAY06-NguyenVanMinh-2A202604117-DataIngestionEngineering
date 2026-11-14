"""Show head and tail of each coherence-audit chunk (to judge if an idea is cut)."""
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
with (REPO_ROOT / "results" / "audit_coherence.csv").open(encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        text = row["chunk_text"]
        print(f"\n### {row['strategy']} {row['document_id']} p{row['page']} len={row['char_len']}")
        print(f"  HEAD: {text[:160].replace(chr(10), ' | ')}")
        print(f"  TAIL: ...{text[-200:].replace(chr(10), ' | ')}")
