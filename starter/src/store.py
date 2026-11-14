"""Qdrant vector store (embedded persistent mode): cosine search + metadata filter."""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from .models import Chunk

NAMESPACE = uuid.UUID("6f1b1a2c-3d4e-5f60-8a9b-0c1d2e3f4a5b")
FILTER_FIELDS = ("year", "primary_category", "document_type", "language")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID so re-ingesting the same chunk overwrites, never duplicates."""
    return str(uuid.uuid5(NAMESPACE, chunk_id))


class QdrantStore:
    def __init__(self, path: Path, collection: str, dimensions: int):
        self.path = Path(path)
        self.collection = collection
        self.dimensions = dimensions
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(self.path))

    def ensure_collection(self, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=self.dimensions, distance=models.Distance.COSINE),
            )
            for field in FILTER_FIELDS:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD
                    if field in ("primary_category", "document_type", "language")
                    else models.PayloadSchemaType.INTEGER,
                )
            return
        info = self.client.get_collection(self.collection)
        size = info.config.params.vectors.size
        if size != self.dimensions:
            raise RuntimeError(
                f"collection '{self.collection}' has dim {size}, embedder produces {self.dimensions}; "
                "re-run with --recreate"
            )

    def existing_ids(self, strategy: str, document_ids: list[str]) -> set[str]:
        """Point IDs already stored for these documents under this strategy."""
        if not document_ids:
            return set()
        found: set[str] = set()
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=models.Filter(must=[
                    models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy)),
                    models.FieldCondition(key="document_id",
                                          match=models.MatchAny(any=document_ids)),
                ]),
                limit=512,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            found.update(str(r.id) for r in records)
            if offset is None:
                break
        return found

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> dict:
        """Write chunks with stable IDs and delete chunks that no longer exist."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for vec in vectors:
            if len(vec) != self.dimensions:
                raise ValueError(f"vector dim {len(vec)} != {self.dimensions}")

        document_ids = sorted({c.document_id for c in chunks})
        stale = self.existing_ids(strategy, document_ids) - {point_id(c.chunk_id) for c in chunks}
        if stale:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=sorted(stale)),
            )

        points = [
            models.PointStruct(id=point_id(chunk.chunk_id), vector=vec, payload=chunk.payload())
            for chunk, vec in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 256):
            self.client.upsert(collection_name=self.collection, points=points[i:i + 256])
        return {"strategy": strategy, "upserted": len(points), "deleted_stale": len(stale)}

    def build_filter(self, strategy: str, filters: dict[str, Any] | None) -> models.Filter:
        must = [models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))]
        for key, value in (filters or {}).items():
            if key not in FILTER_FIELDS:
                raise ValueError(f"unsupported filter field '{key}'; allowed: {FILTER_FIELDS}")
            must.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
        return models.Filter(must=must)

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Cosine top-k inside one strategy, optionally restricted by metadata."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=self.build_filter(strategy, filters),
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "rank": rank,
                "score": round(float(p.score), 6),
                "chunk_id": p.payload.get("chunk_id"),
                "document_id": p.payload.get("document_id"),
                "page": p.payload.get("page"),
                "section": p.payload.get("section"),
                "year": p.payload.get("year"),
                "primary_category": p.payload.get("primary_category"),
                "source": p.payload.get("source"),
                "text": p.payload.get("text", ""),
            }
            for rank, p in enumerate(response.points, start=1)
        ]

    def count(self, strategy: str | None = None) -> int:
        count_filter = None
        if strategy:
            count_filter = models.Filter(must=[
                models.FieldCondition(key="strategy", match=models.MatchValue(value=strategy))])
        return self.client.count(self.collection, count_filter=count_filter,
                                 exact=True).count

    def close(self) -> None:
        self.client.close()
