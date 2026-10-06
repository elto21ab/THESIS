"""retrieve(query, filters, k) -> ranked chunks. Overlapping spans from the same thread
are collapsed at retrieval time (the 40% overlap is for recall, not for duplicate hits)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunk.base import Chunk
from .embed.base import Embedder
from .index import Index


@dataclass(slots=True)
class Hit:
    score: float
    chunk: Chunk


def retrieve(index: Index, query: str, embedder: Embedder, *, k: int = 8,
             platform: str | None = None, partner: str | None = None,
             merge_overlap: bool = True) -> list[Hit]:
    if not index.chunks:
        return []
    q = embedder.embed([query], is_query=True)[0]
    scores = index.vectors @ q
    order = np.argsort(-scores)
    hits: list[Hit] = []
    for idx in order:
        c = index.chunks[int(idx)]
        if platform and c.platform != platform:
            continue
        if partner and c.partner != partner:
            continue
        if merge_overlap and any(h.chunk.platform == c.platform and h.chunk.partner == c.partner
                                 and h.chunk.ts0 <= c.ts1 and c.ts0 <= h.chunk.ts1 for h in hits):
            continue
        hits.append(Hit(float(scores[int(idx)]), c))
        if len(hits) >= k:
            break
    return hits
