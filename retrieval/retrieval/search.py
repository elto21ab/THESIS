"""Retrieval: one semantic contract, two backends.

  vector  — cosine top-k over `vectors.npy` (one matmul against every chunk).
  lexical — regex scan over the blob (exact names, numbers, dates, quoted phrases; no model).

Both share the same filters and the same **overlap rule**: consecutive overlapping chunks of one
thread are *paired into one item*, never dropped and never shown twice. Two probes (or two
neighbours of one query) landing on adjacent chunks is evidence that the whole region is relevant, so
the item becomes the union span with the repeated messages counted once — a repeated message would
read as emphasis the donor never gave. Grouping is by chunk count (2 chunks → 1 item, 3 → 2, 4 → 2),
never by size, so items stay comparable for scoring.

A merged hit exposes `members` (chunk ids), the union span `i0..i1`, the deduped `text`, and an
estimated token count (exact per-message tokens aren't stored; the estimate uses mean tokens/msg).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .chunk import Chunk
from .index import Index, read_blob
from .models import Embedder


@dataclass(slots=True)
class Hit:
    """One retrieved region: 1..n chunks of a single thread, unioned and deduplicated.
    `lines` maps absolute message index → rendered line, so a merge is a dict update and duplicate
    overlap messages are counted once by construction (no text matching involved)."""
    score: float
    members: list[Chunk]
    lines: dict[int, str]
    i0: int
    i1: int
    n_msgs: int
    n_tokens: int

    @property
    def chunk(self) -> Chunk:
        """Primary member — what an unmerged hit has always been."""
        return self.members[0]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.members]

    @property
    def platform(self) -> str:
        return self.members[0].platform

    @property
    def partner(self) -> str:
        return self.members[0].partner

    @property
    def text(self) -> str:
        head = min(self.members, key=lambda c: c.i0).header  # i0's day, if the union spans two
        return head + "\n" + "\n".join(self.lines[i] for i in sorted(self.lines))

    def add(self, c: Chunk, score: float) -> None:
        self.members.append(c)
        self.score = max(self.score, score)
        for i, ln in zip(range(c.i0, c.i1), c.msg_lines()):
            self.lines.setdefault(i, ln)
        self.i0, self.i1 = min(self.i0, c.i0), max(self.i1, c.i1)
        self.n_msgs = len(self.lines)
        # exact per-message token counts aren't stored; estimate from the members' mean tokens/msg
        tok_per_msg = (sum(m.n_tokens for m in self.members)
                       / max(1, sum(m.n_msgs for m in self.members)))
        self.n_tokens = round(self.n_msgs * tok_per_msg)


def _from_chunk(c: Chunk, score: float) -> Hit:
    lines = {i: ln for i, ln in zip(range(c.i0, c.i1), c.msg_lines())}
    return Hit(score, [c], lines, c.i0, c.i1, c.n_msgs, c.n_tokens)


def _keep(c: Chunk, platform: str | None, partner: str | None,
          since: str | None, until: str | None) -> bool:
    if platform and c.platform != platform:
        return False
    if partner and c.partner != partner:
        return False
    if since and c.day < since:
        return False
    if until and c.day > until:
        return False
    return True


def merge_top(candidates: Iterable[tuple[float, Chunk]], *, k: int,
              max_region_chunks: int = 2) -> list[Hit]:
    """Greedy fill of `k` items from a *ranked* candidate list, **pairing** overlapping neighbours.

    Consecutive overlapping chunks are grouped in pairs: 2 chunks → 1 item, 3 → 2 (one pair + the
    single), 4 → 2 pairs. The grouping is by chunk count, never by size, so **no message is ever
    presented twice** — a repeated message reads as emphasis the donor never gave. Merging frees
    slots, so the fill continues further down the ranked list until `k` items exist.

    Order matters: this runs *after* reranking (see `search_any`), so the cross-encoder scores
    uniform, unmerged chunks — its native unit — rather than unions of varying length."""
    hits: list[Hit] = []
    for score, c in candidates:
        target = next((h for h in hits if h.platform == c.platform and h.partner == c.partner
                       and h.i0 <= c.i1 and c.i0 <= h.i1), None)
        if target is not None and len(target.members) >= max_region_chunks:
            target = None  # pair complete: this chunk starts the next item
        if target is None:
            hits.append(_from_chunk(c, score))
            if len(hits) >= k:
                break
            continue
        target.add(c, score)
    return hits


# ── vector backend ────────────────────────────────────────────────────────────
def search_vector(index: Index, query: str, embedder: Embedder, *, k: int = 8,
                  platform: str | None = None, partner: str | None = None,
                  since: str | None = None, until: str | None = None,
                  max_region_chunks: int = 2) -> list[Hit]:
    """`since`/`until` are ISO dates (YYYY-MM-DD) on the chunk's logical day, inclusive."""
    if not index.chunks or index.vectors is None:
        return []
    q = embedder.embed([query], is_query=True)[0]
    scores = index.vectors @ q
    order = np.argsort(-scores)
    return _rank([(float(scores[int(i)]), index.chunks[int(i)]) for i in order], k=k,
                 platform=platform, partner=partner, since=since, until=until,
                 max_region_chunks=max_region_chunks)


# ── lexical backend ───────────────────────────────────────────────────────────
def compile_query(pattern: str) -> re.Pattern:
    """A raw regex if it compiles; else the literal phrase (so a stray `(` or `?` in a
    natural-language query can't blow up the search)."""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(pattern), re.IGNORECASE)


def search_lexical(index: Index, query: str, *, k: int = 8, max_matches: int = 20_000,
                   platform: str | None = None, partner: str | None = None,
                   since: str | None = None, until: str | None = None,
                   max_region_chunks: int = 2) -> list[Hit]:
    """Rank chunks by number of regex matches (earlier position breaks ties), then merge."""
    text, offs = read_blob(index.root) if index.root else (None, None)
    if text is None:
        return []
    rx = compile_query(query)
    counts: dict[int, int] = {}
    for n, m in enumerate(rx.finditer(text)):
        if n >= max_matches:
            break
        i = int(np.searchsorted(offs, m.start(), side="right") - 1)
        counts[i] = counts.get(i, 0) + 1
    ordered = sorted(((float(n), index.chunks[i]) for i, n in counts.items()),
                     key=lambda t: (-t[0], t[1].seq))
    return _rank(ordered, k=k, platform=platform, partner=partner, since=since, until=until,
                 max_region_chunks=max_region_chunks)


# ── dispatch ──────────────────────────────────────────────────────────────────
def search_any(index: Index, query: str, embedder: Embedder | None = None, *, k: int = 8,
               platform: str | None = None, partner: str | None = None,
               since: str | None = None, until: str | None = None,
               max_region_chunks: int = 2, reranker=None, pool: int = 50) -> list[Hit]:
    """The whole retrieval pipeline in one call — the only entry point any caller needs:

        candidates (vector | regex, filtered, pool)  →  [rerank]  →  merge_top(k)

    Reranking sits *between* scoring and merging: the cross-encoder judges uniform, unmerged
    chunks, and merging then collapses near-duplicates so the k returned items stay comparable in
    size and free of repeated messages."""
    cands = candidates(index, query, embedder, pool=pool, platform=platform, partner=partner,
                       since=since, until=until)
    cands = rerank_candidates(cands, query, reranker)
    return merge_top(cands, k=k, max_region_chunks=max_region_chunks)


def _filters(platform, partner, since, until) -> dict:
    return dict(platform=platform, partner=partner, since=since, until=until)


# ── stage 1: score candidates (no merging) ────────────────────────────────────
def candidates(index: Index, query: str, embedder: Embedder | None = None, *, pool: int = 50,
               platform: str | None = None, partner: str | None = None,
               since: str | None = None, until: str | None = None) -> list[tuple[float, Chunk]]:
    """Ranked candidates, unmerged: everything that passes the filters, best first, capped at
    `pool`. This is what a reranker gets to see."""
    f = _filters(platform, partner, since, until)
    if index.kind == "lexical":
        out: list[tuple[float, Chunk]] = []
        text, offs = read_blob(index.root) if index.root else (None, None)
        if text is None:
            return []
        rx = compile_query(query)
        counts: dict[int, int] = {}
        for n, m in enumerate(rx.finditer(text)):
            if n >= 20_000:
                break
            i = int(np.searchsorted(offs, m.start(), side="right") - 1)
            counts[i] = counts.get(i, 0) + 1
        for i, n in counts.items():
            c = index.chunks[i]
            if _keep(c, **f):
                out.append((float(n), c))
        out.sort(key=lambda t: (-t[0], t[1].seq))
        return out[:pool]
    if not index.chunks or index.vectors is None:
        return []
    q = embedder.embed([query], is_query=True)[0]
    scores = index.vectors @ q
    out = [(float(scores[i]), c) for i, c in enumerate(index.chunks) if _keep(c, **f)]
    out.sort(key=lambda t: -t[0])
    return out[:pool]


def rerank_candidates(cands: list[tuple[float, Chunk]], query: str, reranker) -> list[tuple[float, Chunk]]:
    """Replace retrieval scores with cross-encoder scores (the reranker's own ranking)."""
    if not cands or reranker is None:
        return cands
    res = reranker.rerank(query, [c.text for _, c in cands])
    by_idx = {i: s for i, s in res}
    out = [(by_idx.get(i, float("-inf")), c) for i, (_, c) in enumerate(cands)]
    out.sort(key=lambda t: -t[0])
    return out
