"""Count AI-style tells in a text file (byte-safe, unlike grep bracket classes)."""
import collections
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

# (label, list of substrings to count)
CHECKS = [
    ("em dash (U+2014)", ["\u2014"]),
    ("en dash (U+2013)", ["\u2013"]),
    ("curly double quote", ["\u201c", "\u201d"]),
    ("curly single quote", ["\u2018", "\u2019"]),
    ("bold span (**...**)", ["**"]),
]
for label, needles in CHECKS:
    total = sum(text.count(n) for n in needles)
    print(f"{label:24s} {total}")

words = text.split()
print(f"\nwords: {len(words)}")
print(f"lines: {len(text.splitlines())}")

# Sentence-length spread: AI text tends to be uniform.
import re  # noqa: E402

sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
lengths = [len(s.split()) for s in sentences]
if lengths:
    buckets = collections.Counter(min(n // 10 * 10, 60) for n in lengths)
    print(f"sentences: {len(lengths)}, mean words {sum(lengths)/len(lengths):.1f}, "
          f"min {min(lengths)}, max {max(lengths)}")
    print("length buckets (words//10*10):", dict(sorted(buckets.items())))
