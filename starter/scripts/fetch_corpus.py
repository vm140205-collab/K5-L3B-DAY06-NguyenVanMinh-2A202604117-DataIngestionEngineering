"""Fetch the Day 06 corpus from arXiv and build a verifiable manifest.

Corpus: 14 arXiv preprints on reinforcement learning and optimal control, all
published under CC BY 4.0 (verified from each abs page before download).

Outputs (paths relative to repo root):
  data/corpus/<arxiv_id>.pdf   raw PDFs (git-ignored, re-downloadable)
  data/pages.jsonl             one JSON object per physical page, normalized text
  data/manifest.json           per-document provenance, license, page accounting

Usage:
  python starter/scripts/fetch_corpus.py            # download missing + rebuild
  python starter/scripts/fetch_corpus.py --verify   # re-check license only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
PAGES_PATH = DATA_DIR / "pages.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# Minimum characters of normalized, non-bibliography text for a physical page to
# count as a "content page" (GUIDE.md section 1).
MIN_CONTENT_CHARS = 1000

# Fixed candidate list: (arxiv_id, primary_category, short title). License is
# verified at fetch time; the run aborts if a paper is not CC BY 4.0.
CANDIDATES: list[tuple[str, str, str]] = [
    ("2004.01625v2", "eess.SY", "Forward-looking persistent excitation in MPC"),
    ("2104.10383v1", "eess.SY", "Stochastic MPC for linear systems with unbounded noise"),
    ("2404.02429v1", "cs.LG", "AD4RL: autonomous driving benchmarks for offline RL"),
    ("2410.14069v1", "cs.LG", "Optimal transport in offline RL"),
    ("2305.03360v1", "cs.LG", "Survey on offline model-based RL"),
    ("2506.17155v2", "cs.LG", "Sparse-Reg sample complexity"),
    ("2308.14897v1", "cs.LG", "Double policy estimation variance reduction"),
    ("2503.07988v1", "cs.LG", "Provable zero-shot generalization in offline RL"),
    ("2505.22866v1", "cs.LG", "Shortcut models for offline RL"),
    ("2511.10383v2", "stat.ML", "Infinitesimal operator world models"),
    ("2206.06841v1", "cs.LG", "Distributional risk-averse robust RL"),
    ("2303.13117v1", "math.OC", "RLOR: deep RL for operations research"),
    ("2410.17904v1", "cs.LG", "RL under latent dynamics"),
    ("2308.02391v1", "cs.LG", "Optimal admission control in queueing networks"),
]

UA = {"User-Agent": "day6-lab-corpus/1.0 (student assignment; contact: repo owner)"}
REF_HEADING = re.compile(r"^\s*(?:\d+\.?\s*)?(references|bibliography|works cited)\s*$", re.I)
# Start of the section that follows the bibliography (appendix / acknowledgement).
# PDF text extraction often puts the letter on its own line: "A" then "Limitations".
APPENDIX_HEADING = re.compile(
    r"^(?:[A-Z]\.?\d*$|[A-Z]\.\s+\S|[A-Z]\s+[A-Z][a-z]+|Appendix\b|"
    r"ACKNOWLEDG(?:E)?MENTS?$|Supplementary\s+Materials$)"
)


def http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def verify_license(arxiv_id: str) -> str:
    """Return the license string declared on the arXiv abstract page."""
    html = http_get(f"https://arxiv.org/abs/{arxiv_id}").decode("utf-8", "replace")
    m = re.search(r"creativecommons\.org/licenses/([a-z\-]+)/([\d.]+)", html)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    if "nonexclusive-distrib" in html:
        return "arXiv-non-exclusive"
    return "unknown"


def normalize_text(text: str) -> str:
    """NFKC-normalize, repair hyphenated line breaks, collapse intra-line whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")             # soft hyphen
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate across line breaks
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if line:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def strip_running_headers(pages: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    """Drop blocks/lines repeated on >=50% of pages (running headers, footers, page numbers)."""
    if len(pages) < 4:
        return pages, []
    counts: Counter[str] = Counter()
    for blocks in pages:
        seen = set()
        for block in blocks:
            for line in block.splitlines():
                if line and line not in seen:
                    seen.add(line)
                    counts[line] += 1
    threshold = max(4, int(0.5 * len(pages)))
    repeated = {ln for ln, n in counts.items() if n >= threshold and len(ln) <= 120}
    cleaned: list[list[str]] = []
    for blocks in pages:
        page_blocks = []
        for block in blocks:
            kept = "\n".join(ln for ln in block.splitlines() if ln not in repeated).strip()
            if kept:
                page_blocks.append(kept)
        cleaned.append(page_blocks)
    return cleaned, sorted(repeated)


def find_bibliography_range(pages: list[str]) -> tuple[int, int] | None:
    """Return the inclusive [start, end] page indexes of the bibliography block.

    The block starts on the page carrying the References heading and ends on the
    page before the next appendix/acknowledgement heading (or the last page).
    """
    start = None
    for i, text in enumerate(pages):
        if any(REF_HEADING.match(ln) for ln in text.splitlines()):
            start = i
            break
    if start is None:
        return None
    for j in range(start + 1, len(pages)):
        first = next((ln.strip() for ln in pages[j].splitlines() if ln.strip()), "")
        if APPENDIX_HEADING.match(first):
            return start, j - 1
    return start, len(pages) - 1


def classify_pages(pages: list[str]) -> list[tuple[str, int]]:
    """Return [(page_kind, body_chars)] per page. Kinds: content | bibliography | sparse."""
    bib = find_bibliography_range(pages)
    out = []
    for i, text in enumerate(pages):
        if bib and bib[0] <= i <= bib[1]:
            out.append(("bibliography", 0))
            continue
        chars = len(text)
        out.append(("content", chars) if chars >= MIN_CONTENT_CHARS else ("sparse", chars))
    return out


def extract_pages(pdf_path: Path) -> list[list[str]]:
    """Extract layout blocks per page.

    Two-column papers have no blank lines in the flat text, so paragraph structure
    only survives through layout blocks; chunking strategy B depends on this.
    """
    doc = pymupdf.open(pdf_path)
    try:
        pages: list[list[str]] = []
        for page in doc:
            blocks = []
            for block in page.get_text("blocks"):
                if block[6] != 0:  # skip image blocks
                    continue
                text = normalize_text(block[4])
                if text:
                    blocks.append(text)
            pages.append(blocks)
        return pages
    finally:
        doc.close()


def build_document(arxiv_id: str, category: str, note: str) -> tuple[dict, list[dict]]:
    pdf_path = CORPUS_DIR / f"{arxiv_id}.pdf"
    if not pdf_path.exists():
        pdf_path.write_bytes(http_get(f"https://arxiv.org/pdf/{arxiv_id}"))
        time.sleep(1.0)  # be polite to arXiv

    raw_pages = extract_pages(pdf_path)
    block_pages, dropped_headers = strip_running_headers(raw_pages)
    pages = ["\n\n".join(blocks) for blocks in block_pages]

    page_rows = []
    content_pages = []
    for i, (text, (kind, chars)) in enumerate(zip(pages, classify_pages(pages))):
        if kind == "content":
            content_pages.append(i + 1)
        page_rows.append({
            "document_id": arxiv_id,
            "page": i + 1,
            "text": text,
            "blocks": block_pages[i],
            "body_chars": chars,
            "page_kind": kind,
            "is_content_page": kind == "content",
        })

    meta = {
        "source_id": arxiv_id,
        "title": note,
        "source_path_or_url": f"https://arxiv.org/abs/{arxiv_id}",
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "license_or_permission": verify_license(arxiv_id),
        "accessed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "primary_category": category,
        "year": int(arxiv_id[:2]) + 2000,
        "document_type": "preprint",
        "language": "en",
        "physical_pages": len(pages),
        "content_pages": content_pages,
        "content_page_count": len(content_pages),
        "content_chars": sum(r["body_chars"] for r in page_rows),
        "total_chars": sum(len(p) for p in pages),
        "dropped_running_headers": dropped_headers,
        "sha256_pdf": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        "bibliography_pages": (
            list(range(find_bibliography_range(pages)[0] + 1, find_bibliography_range(pages)[1] + 2))
            if find_bibliography_range(pages) else []
        ),
        "page_rule": (
            f"content page = physical page with >= {MIN_CONTENT_CHARS} characters of normalized "
            "text, excluding the bibliography block (References heading up to the next "
            "appendix/acknowledgement heading) and blank/figure-only pages"
        ),
    }
    return meta, page_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="only re-verify licenses")
    args = ap.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    if args.verify:
        bad = 0
        for aid, _, _ in CANDIDATES:
            lic = verify_license(aid)
            ok = lic == "CC BY 4.0"
            bad += 0 if ok else 1
            print(f"{'OK ' if ok else 'BAD'} {aid} {lic}")
        return 1 if bad else 0

    manifest = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "corpus_topic": "reinforcement learning and optimal control (arXiv preprints)",
        "documents": [],
    }
    all_pages: list[dict] = []
    for aid, cat, note in CANDIDATES:
        meta, rows = build_document(aid, cat, note)
        if meta["license_or_permission"] != "CC BY 4.0":
            print(f"ABORT: {aid} license is {meta['license_or_permission']}", file=sys.stderr)
            return 2
        manifest["documents"].append(meta)
        all_pages.extend(rows)
        print(f"{aid}  pages={meta['physical_pages']:3d}  content={meta['content_page_count']:3d}  "
              f"chars={meta['content_chars']:7d}  {meta['title'][:55]}")

    manifest["totals"] = {
        "documents": len(manifest["documents"]),
        "physical_pages": sum(d["physical_pages"] for d in manifest["documents"]),
        "content_pages": sum(d["content_page_count"] for d in manifest["documents"]),
        "content_chars": sum(d["content_chars"] for d in manifest["documents"]),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with PAGES_PATH.open("w", encoding="utf-8") as fh:
        for row in all_pages:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    t = manifest["totals"]
    print(f"\nTOTAL documents={t['documents']} physical_pages={t['physical_pages']} "
          f"content_pages={t['content_pages']} content_chars={t['content_chars']}")
    print(f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} and {PAGES_PATH.relative_to(REPO_ROOT)}")
    return 0 if t["content_pages"] >= 30 else 3


if __name__ == "__main__":
    raise SystemExit(main())
