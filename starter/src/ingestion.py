"""Read the normalized corpus and expose page-level structure to the chunkers."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from .config import MANIFEST_PATH, PAGES_PATH
from .models import Block, DocumentPage

# "3 Experiments", "3.1 Setup", "A Proofs", "B.2 Lemma" - short lines that start a section.
HEADING_RE = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.?\d*)\s+[A-Z][A-Za-z0-9 ,:\-()]{2,70}$")
# PDF extraction often puts the section number on its own line: "1" then "Introduction".
NUMBER_ONLY_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[A-Z]\.?\d*)\.?$")
HEADING_TITLE_RE = re.compile(r"^[A-Z][A-Za-z0-9 ,:\-()/&']{2,70}$")


def normalize_text(text: str) -> str:
    """NFKC-normalize, repair hyphenated line breaks, collapse intra-line whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    out: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", raw).strip()
        if line:
            out.append(line)
        elif out and out[-1] != "":
            out.append("")
    return "\n".join(out).strip()


def load_manifest(path: Path = MANIFEST_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus(pages_path: Path = PAGES_PATH, manifest_path: Path = MANIFEST_PATH
                ) -> tuple[dict, list[DocumentPage]]:
    """Load committed page text plus per-document metadata.

    Only content pages are returned; bibliography and figure-only pages are kept
    in data/pages.jsonl for auditing but never chunked.
    """
    manifest = load_manifest(manifest_path)
    docs = {d["source_id"]: d for d in manifest["documents"]}
    pages: list[DocumentPage] = []
    with pages_path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not row.get("is_content_page"):
                continue
            meta = docs[row["document_id"]]
            pages.append(DocumentPage(
                document_id=row["document_id"],
                page=row["page"],
                text=row["text"],
                blocks=tuple(row.get("blocks") or ()),
                doc_meta=meta,
            ))
    return manifest, pages


def split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split page text on blank lines; return [(char_offset, paragraph_text)]."""
    out: list[tuple[int, str]] = []
    cursor = 0
    for block in re.split(r"\n\s*\n", text):
        stripped = block.strip()
        if stripped:
            start = text.find(stripped, cursor)
            out.append((start if start >= 0 else cursor, stripped))
            cursor = (start if start >= 0 else cursor) + len(stripped)
    return out


def heading_of(paragraph: str) -> str | None:
    """Return the section heading carried by this paragraph, or None.

    Handles both "3.1 Setup" on one line and the split form where the number sits
    alone on the first line and the title on the next.
    """
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    if len(first) > 80:
        return None
    if HEADING_RE.match(first):
        return first
    if NUMBER_ONLY_RE.match(first) and len(lines) > 1:
        second = lines[1]
        if len(second) <= 70 and HEADING_TITLE_RE.match(second):
            return f"{first} {second}"
    return None


def is_heading(paragraph: str) -> bool:
    return heading_of(paragraph) is not None


def page_paragraphs(page: DocumentPage) -> list[tuple[int, str]]:
    """Paragraphs of a page with their character offsets in the flat page text.

    Uses the layout blocks captured at extraction time; falls back to blank-line
    splitting for pages that have none.
    """
    if not page.blocks:
        return split_paragraphs(page.text)
    out: list[tuple[int, str]] = []
    cursor = 0
    for block in page.blocks:
        start = page.text.find(block, cursor)
        if start < 0:
            start = cursor
        out.append((start, block))
        cursor = start + len(block)
    return out


def build_blocks(page: DocumentPage, carry_section: str = "") -> tuple[list[Block], str]:
    """Tag each paragraph with the section heading it falls under.

    `carry_section` lets a document-level pass inherit the last heading seen on an
    earlier page, so chunks on continuation pages are still labelled.
    """
    blocks: list[Block] = []
    section = carry_section
    for start, para in page_paragraphs(page):
        heading = heading_of(para)
        if heading:
            section = heading
        blocks.append(Block(text=para, start=start, section=section))
    return blocks, section


def section_at(blocks: list[Block], offset: int) -> str:  # noqa: D401
    """Section label in force at a character offset inside the page."""
    current = blocks[0].section if blocks else ""
    for block in blocks:
        if block.start <= offset:
            current = block.section
        else:
            break
    return current


def corpus_stats(manifest: dict, pages: list[DocumentPage]) -> dict:
    """Numbers reported in the README/report; computed, never typed by hand."""
    per_doc = Counter(p.document_id for p in pages)
    return {
        "documents": len(manifest["documents"]),
        "physical_pages": manifest["totals"]["physical_pages"],
        "content_pages": manifest["totals"]["content_pages"],
        "content_chars": manifest["totals"]["content_chars"],
        "content_pages_loaded": len(pages),
        "pages_per_document": dict(sorted(per_doc.items())),
        "year_values": sorted({d["year"] for d in manifest["documents"]}),
        "category_values": sorted({d["primary_category"] for d in manifest["documents"]}),
    }
