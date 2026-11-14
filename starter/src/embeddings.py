"""Real local embedding model (BAAI/bge-small-en-v1.5 via ONNX). No mock fallback.

Provider note: the first choice was the Mistral API (`mistral-embed`). The
available key expired on 2026-11-14, so the pipeline moved to a local ONNX model
instead of silently degrading. Documents use `embed`, queries use `query_embed`,
which applies the retrieval prefix the model expects.
"""
from __future__ import annotations

import time
from typing import Protocol

from fastembed import TextEmbedding

from .config import EMBEDDING_BATCH, EMBEDDING_DIM, EMBEDDING_MODEL


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class FastEmbedEmbedder:
    """Local ONNX embedder; deterministic, free, and independent of any API key."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH):
        started = time.time()
        self.model_name = model_name
        self.dimensions = EMBEDDING_DIM
        self.batch_size = batch_size
        self._model = TextEmbedding(model_name=model_name)
        self.load_seconds = round(time.time() - started, 1)
        self.documents_embedded = 0
        self.queries_embedded = 0
        self.embed_seconds = 0.0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = time.time()
        vectors = [[float(x) for x in vec] for vec in
                   self._model.embed(texts, batch_size=self.batch_size)]
        self.embed_seconds += time.time() - started
        if len(vectors) != len(texts):
            raise RuntimeError(f"model returned {len(vectors)} vectors for {len(texts)} inputs")
        for vec in vectors:
            if len(vec) != self.dimensions:
                raise RuntimeError(f"unexpected dimension {len(vec)} != {self.dimensions}")
        self.documents_embedded += len(vectors)
        return vectors

    def embed_query(self, question: str) -> list[float]:
        started = time.time()
        vector = [float(x) for x in next(iter(self._model.query_embed(question)))]
        self.embed_seconds += time.time() - started
        self.queries_embedded += 1
        return vector

    def usage(self) -> dict:
        return {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "provider": "fastembed 0.9.1 (local ONNX runtime)",
            "documents_embedded": self.documents_embedded,
            "queries_embedded": self.queries_embedded,
            "model_load_seconds": self.load_seconds,
            "embed_seconds": round(self.embed_seconds, 1),
            "cost_usd": 0.0,
        }
