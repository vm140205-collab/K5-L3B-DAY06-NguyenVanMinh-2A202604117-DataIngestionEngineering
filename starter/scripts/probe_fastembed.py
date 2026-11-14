"""Probe: does a local ONNX embedding model load and produce sane similarities?"""
import time

from fastembed import TextEmbedding

t0 = time.time()
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print(f"loaded in {time.time() - t0:.1f}s")

docs = [
    "We propose pessimistic empirical risk minimization and pessimistic proximal policy optimization.",
    "The algorithm learns admission control in partially observable queueing networks.",
    "Sparse-Reg uses sparse regularization to regularize model complexity.",
]
queries = ["Which two meta-algorithms are proposed for zero-shot generalization?"]

doc_vecs = list(model.embed(docs))
query_vec = list(model.embed(queries))[0]
print("dim:", len(query_vec))
import numpy as np  # noqa: E402

for doc, vec in zip(docs, doc_vecs):
    sim = float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec)))
    print(f"{sim:.4f}  {doc[:60]}")
