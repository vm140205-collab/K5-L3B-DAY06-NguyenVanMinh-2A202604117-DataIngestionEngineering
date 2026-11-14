"""Calibrate the bibliography-page rule against the real corpus.

For every page, print how many of its lines look like citation entries under
strict patterns only (no inline markers), so the threshold can be chosen from
data instead of guessed.
"""
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
pages = [json.loads(line) for line in (REPO_ROOT / "data" / "pages.jsonl").open(encoding="utf-8")]

STRICT = (
    re.compile(r"^(?:\d{1,3}\.\s*)?[A-Z][A-Za-z\u00C0-\u024F'\-]+,\s+[A-Z]\."),
    re.compile(r"^(?:\d{1,3}\.\s*)?[A-Z][a-z\u00C0-\u024F'\-]+ [A-Z][A-Za-z\u00C0-\u024F'\-]+.*"
               r"(?:\((?:19|20)\d{2}[a-z]?\)|,\s*(?:19|20)\d{2}[a-z]?\.)"),
    re.compile(r"^\[[A-Za-z0-9]+\]\s*[A-Z]"),
)
HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.I)


def strict_ratio(text: str) -> tuple[float, int, int]:
    lines = [ln for ln in text.splitlines() if ln]
    if not lines:
        return 0.0, 0, 0
    hits = sum(1 for ln in lines if any(p.match(ln) for p in STRICT))
    return hits / len(lines), hits, len(lines)


rows = []
for p in pages:
    ratio, hits, total = strict_ratio(p["text"])
    heading = any(HEADING.match(ln) for ln in p["text"].splitlines())
    rows.append((ratio, hits, total, p["document_id"], p["page"], heading,
                 p["text"].splitlines()[0][:70] if p["text"].strip() else ""))

rows.sort(reverse=True)
print(f"{'ratio':>6} {'hits':>5} {'lines':>5} {'head':>5}  doc/page  first line")
for r in rows[:45]:
    print(f"{r[0]:6.2f} {r[1]:5d} {r[2]:5d} {str(r[5]):>5}  {r[3]}/p{r[4]:<3d} {r[6]}")

print("\n--- distribution of ratio for pages with >=15 lines ---")
buckets = {"0.0": 0, "0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
for ratio, _, total, *_ in rows:
    if total < 15:
        continue
    if ratio == 0:
        buckets["0.0"] += 1
    elif ratio < 0.2:
        buckets["0-0.2"] += 1
    elif ratio < 0.4:
        buckets["0.2-0.4"] += 1
    elif ratio < 0.6:
        buckets["0.4-0.6"] += 1
    elif ratio < 0.8:
        buckets["0.6-0.8"] += 1
    else:
        buckets["0.8-1.0"] += 1
print(buckets)
