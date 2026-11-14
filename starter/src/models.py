"""Shared data contracts; keep traceability from chunk back to source."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DocumentPage:
    """One physical page of one source document, already normalized."""

    document_id: str
    page: int
    text: str
    blocks: tuple[str, ...] = ()
    doc_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return str(self.doc_meta.get("source_path_or_url", ""))

    @property
    def title(self) -> str:
        return str(self.doc_meta.get("title", ""))


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit. `page` is the physical page it starts on."""

    chunk_id: str
    document_id: str
    page: int
    text: str
    strategy: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        """Flat payload written to the vector store."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "page": self.page,
            "text": self.text,
            "strategy": self.strategy,
            **self.metadata,
        }


@dataclass(frozen=True)
class Block:
    """A paragraph inside a page, tagged with the section heading it belongs to."""

    text: str
    start: int
    section: str
