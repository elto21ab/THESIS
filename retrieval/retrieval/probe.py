"""STUB. Push/pack builder over generic retrieve(). Probe source deferred (see PLAN.md)."""
from __future__ import annotations

from .chunk.base import Chunk
from .embed.base import Embedder
from .index import Index
from .retrieve import retrieve


def build_pack(index: Index, probes: list[str], embedder: Embedder, *,
               per_probe: int = 4, budget_tokens: int = 40_000,
               platform: str | None = None, partner: str | None = None) -> list[Chunk]:
    seen: set[str] = set()
    pack: list[Chunk] = []
    used = 0
    for q in probes:
        for hit in retrieve(index, q, embedder, k=per_probe, platform=platform, partner=partner):
            c = hit.chunk
            if c.id in seen:
                continue
            seen.add(c.id)
            if used + c.n_tokens > budget_tokens:
                return pack
            pack.append(c)
            used += c.n_tokens
    return pack
