"""For questions whose evidence never reaches top-20: is it split, or just ranked low?

For each such question, look only at the gold page's own chunks and report
  - whether the evidence quote fits inside a single chunk (split vs intact),
  - the cosine similarity of that chunk vs the best overall hit.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import load_questions, squash  # noqa: E402
from src.chunking import chunk_document  # noqa: E402
from src.config import COLLECTION, EMBEDDING_DIM, QDRANT_PATH, RESULTS_DIR, STRATEGIES  # noqa: E402
from src.embeddings import FastEmbedEmbedder  # noqa: E402
from src.ingestion import load_corpus  # noqa: E402
from src.store import QdrantStore  # noqa: E402

depth = json.loads((RESULTS_DIR / "retrieval_depth.json").read_text(encoding="utf-8"))
never_found = {s: set(v["evidence_never_found"]) for s, v in depth["summary"].items()}

_, pages = load_corpus()
by_doc: dict[str, list] = {}
for page in pages:
    by_doc.setdefault(page.document_id, []).append(page)

questions = {q["id"]: q for q in load_questions()}
embedder = FastEmbedEmbedder()
store = QdrantStore(QDRANT_PATH, COLLECTION, EMBEDDING_DIM)

for strategy in ("A", "B"):
    chunks = []
    for doc_id in sorted(by_doc):
        chunks.extend(chunk_document(by_doc[doc_id], STRATEGIES[strategy]))
    print(f"\n=== strategy {strategy}: {len(never_found[strategy])} questions never found")
    for qid in sorted(never_found[strategy]):
        q = questions[qid]
        gold_chunks = [c for c in chunks
                       if c.document_id == q["gold_source_id"] and c.page in q["gold_pages"]]
        quote = squash(q["evidence_quote"])
        holder = [c for c in gold_chunks if quote in squash(c.text)]
        top = store.search(strategy, embedder.embed_query(q["question"]), 1)[0]
        if holder:
            vecs = np.array(embedder.embed_documents([c.text for c in holder]))
            qv = np.array(embedder.embed_query(q["question"]))
            sims = vecs @ qv / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv))
            best = float(sims.max())
            verdict = f"INTACT in chunk (sim {best:.3f} vs top1 {top['score']:.3f})"
        else:
            verdict = f"SPLIT across chunks ({len(gold_chunks)} chunks on gold page)"
        print(f"  {qid}: {verdict}")
store.close()
