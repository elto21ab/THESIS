"""Run a query set against an index, with or without a reranker, and collect target ranks."""
from __future__ import annotations

from ..embed.base import Embedder
from ..index import Index
from ..retrieve import retrieve
from ..rerank import Reranker
from .queries import SynthQuery
from .metrics import summarize


def evaluate(index: Index, embedder: Embedder, queries: list[SynthQuery], *,
             k: int = 10, candidates: int = 50,
             reranker: Reranker | None = None) -> dict:
    ranks: list[int | None] = []
    for q in queries:
        if reranker:
            hits = retrieve(index, q.text, embedder, k=candidates)
            pool = [h.chunk for h in hits]
            order = reranker.rerank(q.text, [c.text for c in pool], top_k=k)
            ranked = [pool[i] for i, _ in order if 0 <= i < len(pool)]
        else:
            hits = retrieve(index, q.text, embedder, k=k)
            ranked = [h.chunk for h in hits]
        rank = next((i + 1 for i, c in enumerate(ranked) if c.id == q.target), None)
        ranks.append(rank)
    return summarize(ranks)


metrics = summarize
