"""Explain the retrieval numbers: per-question outcome and failure mode."""
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

for strategy in ("A", "B", "C"):
    traces = [json.loads(line) for line in (RESULTS / f"traces_{strategy}.jsonl").open(encoding="utf-8")]
    modes = Counter()
    print(f"\n=== strategy {strategy}")
    for t in traces:
        if t["hit_at_3"]:
            modes["hit@3"] += 1
        elif t["gold_page_in_top3"]:
            modes["gold page in top-3 but evidence split/mismatched"] += 1
        else:
            modes["gold page not in top-3"] += 1
        if not t["hit_at_3"]:
            top = [(h["document_id"], h["page"], h["score"]) for h in t["hits"]]
            print(f"  MISS {t['question_id']} [{t['split']}/{t['type']}] filter={t['filter']} "
                  f"-> {top}")
    print(f"  summary: {dict(modes)}")

print("\n=== filter-question compliance (strategy C)")
traces = [json.loads(line) for line in (RESULTS / "traces_C.jsonl").open(encoding="utf-8")]
for t in traces:
    if not t["filter"]:
        continue
    ok = all(all(h.get(k) == v for k, v in t["filter"].items()) for h in t["hits"])
    print(f"  {t['question_id']} filter={t['filter']} compliant={ok} "
          f"hit@3={t['hit_at_3']} top1={t['hits'][0]['document_id']}p{t['hits'][0]['page']} "
          f"year={t['hits'][0]['year']} cat={t['hits'][0]['primary_category']}")
