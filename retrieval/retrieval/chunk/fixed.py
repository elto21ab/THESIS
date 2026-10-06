"""Fixed token-budget window. Segmentation advances non-overlapping; overlap is added
only as a text prefix from the previous segment (so a hard day/max boundary cannot re-trigger
and produce a cascade of tiny duplicate chunks). Boundaries snap to msg + logical day.
Both-people is a soft extend within the same day."""
from __future__ import annotations

from ..io import Thread
from .base import Chunk, ChunkCfg, Count, make_chunk, prepare


def chunk_thread(thread: Thread, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    lines, toks, senders, days = prepare(thread, count, cfg)
    n = len(lines)
    chunks: list[Chunk] = []
    start = 0
    seq = 0
    while start < n:
        end = start
        tot = 0
        who: set[str] = set()
        day = days[start]
        while end < n:
            if end > start and days[end] != day and tot >= cfg.min:
                break  # day boundary, once the segment carries signal
            if end > start and tot + toks[end] > cfg.max:
                break
            tot += toks[end]
            who.add(senders[end])
            day = days[end]
            end += 1
            if tot >= cfg.target and (not cfg.both_people or len(who) >= 2):
                break
        if cfg.both_people and len(who) < 2 and end < n:
            while end < n and days[end] == day and tot + toks[end] <= cfg.max:
                tot += toks[end]
                who.add(senders[end])
                end += 1
                if len(who) >= 2:
                    break
        # overlap prefix: rewind by ~overlap*tot tokens, clamped to the previous segment
        o0 = start
        back, k = 0.0, start
        while k > 0 and back < cfg.overlap * tot:
            k -= 1
            back += toks[k]
        o0 = k
        chunks.append(make_chunk(thread, seq, o0, end, lines, toks, cfg))
        seq += 1
        start = end  # segmentation is non-overlapping; overlap lives only in the text
    return chunks
