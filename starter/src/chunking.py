"""Fixed-size and structure-aware chunking, both page-scoped.

Chunks never span a page boundary. That keeps `page` metadata exact (so a hit can
be checked against a gold page) at the cost of cutting sentences that continue on
the next page; the trade-off is reported in the analysis.
"""
from __future__ import annotations

import hashlib

from .config import Strategy
from .ingestion import build_blocks, section_at
from .models import Block, Chunk, DocumentPage

SENTENCE_SPLIT = ". "
MIN_CHUNK_CHARS = 120  # shorter fragments (page numbers, stray letters) get merged back


def _chunk_id(strategy: str, document_id: str, page: int, index: int, text: str) -> str:
    raw = f"{strategy}|{document_id}|{page}|{index}|{text[:120]}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _meta(page: DocumentPage, section: str, index: int, strategy: str, text: str) -> dict:
    return {
        "source": page.source,
        "title": page.title,
        "year": page.doc_meta.get("year"),
        "primary_category": page.doc_meta.get("primary_category"),
        "document_type": page.doc_meta.get("document_type"),
        "language": page.doc_meta.get("language"),
        "section": section,
        "chunk_index": index,
        "char_len": len(text),
    }


def _snap(text: str, start: int, end: int) -> int:
    """Move a window end back to the nearest whitespace so words are not cut."""
    if end >= len(text):
        return len(text)
    floor = start + int(0.7 * (end - start))
    cut = text.rfind(" ", floor, end)
    return cut if cut > floor else end


def fixed_size(page: DocumentPage, size: int, overlap: int, strategy: str = "A",
               carry_section: str = "") -> list[Chunk]:
    """Split the page into overlapping character windows, keeping page metadata."""
    text = page.text
    blocks, _ = build_blocks(page, carry_section)
    chunks: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = _snap(text, start, min(start + size, len(text)))
        window = text[start:end].strip()
        if window:
            chunks.append(Chunk(
                chunk_id=_chunk_id(strategy, page.document_id, page.page, index, window),
                document_id=page.document_id,
                page=page.page,
                text=window,
                strategy=strategy,
                metadata=_meta(page, section_at(blocks, start), index, strategy, window),
            ))
            index += 1
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _split_oversized(block: Block, max_size: int) -> list[Block]:
    """Split a paragraph longer than max_size on sentence ends, then hard-split."""
    if len(block.text) <= max_size:
        return [block]
    pieces: list[Block] = []
    buf = ""
    buf_start = block.start
    offset = block.start
    for sentence in block.text.split(SENTENCE_SPLIT):
        piece = sentence + SENTENCE_SPLIT if len(sentence) < len(block.text) else sentence
        while len(piece) > max_size:  # a single monster sentence
            if buf:
                pieces.append(Block(buf.strip(), buf_start, block.section))
                buf = ""
            pieces.append(Block(piece[:max_size].strip(), offset, block.section))
            offset += max_size
            piece = piece[max_size:]
        if len(buf) + len(piece) > max_size and buf:
            pieces.append(Block(buf.strip(), buf_start, block.section))
            buf_start = offset
            buf = ""
        if not buf:
            buf_start = offset
        buf += piece
        offset += len(piece)
    if buf.strip():
        pieces.append(Block(buf.strip(), buf_start, block.section))
    return pieces


def by_paragraph(page: DocumentPage, max_size: int, strategy: str = "B",
                 carry_section: str = "") -> list[Chunk]:
    """Pack whole paragraphs up to max_size; never merge across a section heading."""
    blocks, _ = build_blocks(page, carry_section)
    chunks: list[Chunk] = []
    buf: list[Block] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n\n".join(b.text for b in buf).strip()
        chunks.append(Chunk(
            chunk_id=_chunk_id(strategy, page.document_id, page.page, len(chunks), text),
            document_id=page.document_id,
            page=page.page,
            text=text,
            strategy=strategy,
            metadata=_meta(page, buf[0].section, len(chunks), strategy, text),
        ))
        buf.clear()

    for block in blocks:
        for piece in _split_oversized(block, max_size):
            starts_new_section = bool(buf) and piece.section != buf[-1].section
            too_big = buf and sum(len(b.text) + 2 for b in buf) + len(piece.text) > max_size
            if starts_new_section or too_big:
                flush()
            buf.append(piece)
    flush()
    return chunks


def _join(prev: Chunk, extra: str) -> Chunk:
    text = f"{prev.text}\n\n{extra}".strip()
    return Chunk(
        chunk_id=prev.chunk_id,
        document_id=prev.document_id,
        page=prev.page,
        text=text,
        strategy=prev.strategy,
        metadata={**prev.metadata, "char_len": len(text)},
    )


def _merge_tiny(chunks: list[Chunk], max_size: int) -> list[Chunk]:
    """Fold fragments shorter than MIN_CHUNK_CHARS into a neighbour on the same page.

    Heading-only blocks ("Appendix A", "3 Related Work") are the usual offenders;
    leaving them as standalone chunks would make them unretrievable noise.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        if (merged and len(chunk.text) < MIN_CHUNK_CHARS
                and merged[-1].page == chunk.page
                and len(merged[-1].text) + len(chunk.text) + 2 <= max_size * 1.25):
            merged[-1] = _join(merged[-1], chunk.text)
        else:
            merged.append(chunk)

    out: list[Chunk] = []
    for chunk in merged:
        if (out and len(out[-1].text) < MIN_CHUNK_CHARS and out[-1].page == chunk.page
                and len(out[-1].text) + len(chunk.text) + 2 <= max_size * 1.25):
            out[-1] = _join(out[-1], chunk.text)
        else:
            out.append(chunk)
    return out


def chunk_document(pages: list[DocumentPage], strategy: Strategy) -> list[Chunk]:
    """Chunk every page of one document in page order, carrying the section label."""
    out: list[Chunk] = []
    carry = ""
    for page in pages:
        if strategy.chunker == "fixed":
            page_chunks = fixed_size(page, strategy.size, strategy.overlap, strategy.name, carry)
        else:
            page_chunks = by_paragraph(page, strategy.size, strategy.name, carry)
        out.extend(_merge_tiny(page_chunks, strategy.size))
        for block_section in [c.metadata["section"] for c in page_chunks]:
            if block_section:
                carry = block_section
    return out


def chunk_stats(chunks: list[Chunk], strategy: str) -> dict:
    lengths = sorted(len(c.text) for c in chunks)
    if not lengths:
        return {"strategy": strategy, "chunks": 0}
    n = len(lengths)
    return {
        "strategy": strategy,
        "chunks": n,
        "total_chars": sum(lengths),
        "mean_chars": round(sum(lengths) / n, 1),
        "min_chars": lengths[0],
        "p50_chars": lengths[n // 2],
        "p90_chars": lengths[int(0.9 * (n - 1))],
        "max_chars": lengths[-1],
        "documents": len({c.document_id for c in chunks}),
        "sections": len({c.metadata["section"] for c in chunks}),
    }
