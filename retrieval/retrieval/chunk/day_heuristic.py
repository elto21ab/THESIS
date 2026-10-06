"""Day-first heuristic (from the design notes).

Group messages by logical day. Undersized days merge into the most similar neighbour
(MVP: previous, else next) and repeat until >= min. Oversized days split with the fixed
token window to respect max. Single-party days are flagged and still merged when small.
"""
from __future__ import annotations

from ..io import Thread
from . import fixed
from .base import Chunk, ChunkCfg, Count, make_chunk, prepare


def _groups(days: list[str]) -> list[list[int]]:
    out: list[list[int]] = []
    for idx, d in enumerate(days):
        if out and days[out[-1][-1]] == d:
            out[-1].append(idx)
        elif not out:
            out.append([idx])
        else:
            out.append([idx])
    return out


def chunk_thread(thread: Thread, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    lines, toks, senders, days = prepare(thread, count, cfg)

    def size(g: list[int]) -> int:
        return sum(toks[i] for i in g)

    groups = _groups(days)
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for g in range(len(groups)):
            if size(groups[g]) >= cfg.min:
                continue
            pick = g - 1 if g > 0 else g + 1
            if pick < 0:
                continue
            lo, hi = sorted((g, pick))
            groups[lo:hi + 1] = [sorted(groups[lo] + groups[hi])]
            merged = True
            break

    chunks: list[Chunk] = []
    seq = 0
    for g in groups:
        if size(g) <= cfg.max:
            chunks.append(make_chunk(thread, seq, g[0], g[-1] + 1, lines, toks, cfg))
            seq += 1
            continue
        # oversized day -> hand the span to the fixed window (same cfg, max respected)
        sub = Thread(thread.id, thread.platform, thread.partner, thread.participants,
                     thread.messages[g[0]:g[-1] + 1], thread.sources)
        for c in fixed.chunk_thread(sub, cfg, count):
            chunks.append(make_chunk(thread, seq, g[0] + c.i0, g[0] + c.i1, lines, toks, cfg))
            seq += 1
    return chunks
