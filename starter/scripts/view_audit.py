"""Print an audit CSV for manual labelling.

    python starter/scripts/view_audit.py coherence [chars]
    python starter/scripts/view_audit.py grounding [chars]
"""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
kind = sys.argv[1] if len(sys.argv) > 1 else "coherence"
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 700
path = REPO_ROOT / "results" / f"audit_{kind}.csv"

with path.open(encoding="utf-8", newline="") as fh:
    for row in csv.DictReader(fh):
        if kind == "coherence":
            print(f"\n### {row['strategy']} {row['document_id']} p{row['page']} "
                  f"section={row['section']!r} len={row['char_len']}")
            print(row["chunk_text"][:limit])
        else:
            print(f"\n### {row['strategy']} {row['question_id']} evidence_at="
                  f"{row['evidence_present_at_rank']} top1={row['top1']}")
            print(f"Q: {row['question']}")
            print(f"A: {row['expected_answer']}")
            print(f"EV: {row['evidence_quote']}")
            for rank in (1, 2, 3):
                print(f"  [{rank}] {row[f'top{rank}_excerpt'][:limit]}")
