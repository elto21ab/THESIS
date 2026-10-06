"""Synthetic queries: no labels exist yet.

A query is the text of one real message, stripped of its timestamp/sender; the target is the chunk
that message lives in. Self-retrieval, so it measures chunk coherence and ranking sanity (and,
with the reranker, whether it can re-order a plausible candidate set) — not open-domain recall.
Replace with hand-written survey-style probes when the instrument battery is fixed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..chunk.base import Chunk


@dataclass(slots=True)
class SynthQuery:
    text: str
    target: str       # chunk id
    platform: str
    partner: str


def _body(line: str) -> str:
    at = line.find(": ")
    return line[at + 2:] if line.startswith("[") and at != -1 else ""


def synthetic(chunks: list[Chunk], *, n: int = 100, seed: int = 0,
              min_words: int = 6) -> list[SynthQuery]:
    rng = random.Random(seed)
    pool = [c for c in chunks if c.n_msgs >= 2]
    rng.shuffle(pool)
    out: list[SynthQuery] = []
    for c in pool:
        lines = [ln for ln in c.text.split("\n") if ln.startswith("[")]
        rng.shuffle(lines)
        for ln in lines:
            t = _body(ln)
            if len(t.split()) >= min_words:
                out.append(SynthQuery(t, c.id, c.platform, c.partner))
                break
        if len(out) >= n:
            break
    return out
