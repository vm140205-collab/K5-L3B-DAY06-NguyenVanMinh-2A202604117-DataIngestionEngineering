"""Apply the manual audit labels (author-assigned) and summarise the five dimensions.

Labels are written here explicitly, with the reason for each score, so the audit is
reproducible and reviewable instead of living in the report prose only.
Grader = the author (bias risk is stated in the report).
"""
import csv
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"

COHERENCE_REASON_A = ("cua so 800 ky tu bat dau o tieu de/tac gia va cat giua cau cua abstract; "
                      "hieu duoc nhung y bi cut")
COHERENCE_LABELS = {
    ("A", "2004.01625v2"): (1, COHERENCE_REASON_A),
    ("A", "2104.10383v1"): (1, COHERENCE_REASON_A),
    ("A", "2206.06841v1"): (1, COHERENCE_REASON_A),
    ("A", "2303.13117v1"): (1, COHERENCE_REASON_A),
    ("A", "2305.03360v1"): (1, COHERENCE_REASON_A),
    ("A", "2308.02391v1"): (1, COHERENCE_REASON_A),
    ("A", "2308.14897v1"): (1, COHERENCE_REASON_A),
    ("A", "2404.02429v1"): (1, COHERENCE_REASON_A),
    ("A", "2410.14069v1"): (1, COHERENCE_REASON_A),
    ("A", "2410.17904v1"): (1, COHERENCE_REASON_A),
    ("B", "2004.01625v2"): (2, "tron abstract, ket thuc o cau hoan chinh"),
    ("B", "2104.10383v1"): (1, "chi co tieu de + tac gia (154 ky tu), khong co noi dung"),
    ("B", "2206.06841v1"): (2, "tron abstract, ket thuc o cau hoan chinh"),
    ("B", "2303.13117v1"): (1, "tieu de + tac gia + chu 'ABSTRACT', chua vao noi dung"),
    ("B", "2305.03360v1"): (1, "tieu de + tac gia + chu 'Abstract', chua vao noi dung"),
    ("B", "2308.02391v1"): (1, "cat giua cau o cuoi chunk ('we show that our algorithm has')"),
    ("B", "2308.14897v1"): (1, "tieu de + tac gia + chu 'Abstract', chua vao noi dung"),
    ("B", "2404.02429v1"): (2, "tron abstract, ket thuc o URL du an"),
    ("B", "2410.14069v1"): (2, "tron abstract, ket thuc o cau hoan chinh"),
    ("B", "2410.17904v1"): (2, "ket thuc o cau hoan chinh ('poorly understood.')"),
}

GROUNDING_LABELS = {
    ("A", "q16"): (2, "evidence o rank 1"),
    ("A", "q17"): (2, "evidence o rank 1"),
    ("A", "q18"): (1, "top-3 cung tai lieu va lien quan (regularization, hieu qua) nhung thieu cau "
                       "key insight ve self-consistency"),
    ("A", "q19"): (2, "evidence o rank 3"),
    ("A", "q20"): (2, "evidence o rank 3"),
    ("B", "q16"): (2, "evidence o rank 1"),
    ("B", "q17"): (2, "evidence o rank 1"),
    ("B", "q18"): (1, "top-3 cung tai lieu, noi ve scaling/one-stage training nhung thieu cau "
                       "key insight ve self-consistency"),
    ("B", "q19"): (2, "evidence o rank 3"),
    ("B", "q20"): (2, "evidence khong khop nguyen van nhung top-1 neu dung 'sample complexity "
                       "bounds under latent pushforward coverability (Theorem 3.2)' -> du tra loi"),
    ("C", "q16"): (2, "evidence o rank 1"),
    ("C", "q17"): (2, "evidence o rank 1"),
    ("C", "q18"): (1, "top-3 cung tai lieu, noi ve scaling/one-stage training nhung thieu cau "
                       "key insight ve self-consistency"),
    ("C", "q19"): (2, "evidence o rank 3"),
    ("C", "q20"): (2, "evidence khong khop nguyen van nhung top-1 neu dung y (Theorem 3.2)"),
}


def label_coherence() -> dict:
    path = RESULTS / "audit_coherence.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        score, reason = COHERENCE_LABELS[(row["strategy"], row["document_id"])]
        row["coherence_0_1_2"] = score
        row["reason"] = reason
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for strategy in ("A", "B"):
        subset = [r for r in rows if r["strategy"] == strategy]
        scores = [int(r["coherence_0_1_2"]) for r in subset]
        summary[strategy] = {"n": len(scores), "mean": round(sum(scores) / len(scores), 2),
                             "distribution": dict(sorted(Counter(scores).items()))}
    return summary


def label_grounding() -> dict:
    path = RESULTS / "audit_grounding.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    for row in rows:
        score, reason = GROUNDING_LABELS[(row["strategy"], row["question_id"])]
        row["grounding_0_1_2"] = score
        row["reason"] = reason
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for strategy in ("A", "B", "C"):
        subset = [r for r in rows if r["strategy"] == strategy]
        scores = [int(r["grounding_0_1_2"]) for r in subset]
        summary[strategy] = {"n": len(scores), "mean": round(sum(scores) / len(scores), 2),
                             "distribution": dict(sorted(Counter(scores).items()))}
    return summary


def metadata_summary() -> dict:
    rows = list(csv.DictReader((RESULTS / "audit_metadata.csv").open(encoding="utf-8", newline="")))
    out = {}
    for strategy in ("A", "B", "C"):
        subset = [r for r in rows if r["strategy"] == strategy]
        ok = sum(1 for r in subset
                 if r["text_found_on_claimed_page"] == "True"
                 and r["source_matches_manifest"] == "True"
                 and r["year_matches_manifest"] == "True"
                 and r["category_matches_manifest"] == "True")
        filt = [r for r in subset if r["filter_required"] == "True"]
        filt_ok = sum(1 for r in filt if r["filter_satisfied"] == "True")
        out[strategy] = {"records_checked": len(subset), "records_correct": ok,
                         "filter_records": len(filt), "filter_records_compliant": filt_ok}
    return out


summary = {
    "coherence": label_coherence(),
    "grounding": label_grounding(),
    "metadata": metadata_summary(),
    "grader": "author (self-assessment); sample sizes 10 positions / 5 test questions",
}
(RESULTS / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
