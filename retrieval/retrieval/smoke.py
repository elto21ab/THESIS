"""Smoke test: hand-written (query → expected) pairs over a real index.

A case is `{query, partner?, day?, contains?, rank_max?}`. `contains` is a substring that must appear
in the *rendered text of one hit* — the union of a merged region, truncated at `max_chars`, so it is
a check on what the model would read, not on what the index stores. `rank_max` is how deep the first
matching hit is allowed to sit (default: k); without it, a case that slides from rank 1 to rank 4
still passes and the regression is invisible.

`derive()` writes the expectations from the index itself: find the chunk that holds the string, record
its partner and the rank at which the query currently reaches it (plus one, as headroom). A case
derived that way cannot be stale against a rebuild — which is what happened when the token ratio
changed and every hand-written case silently kept its old chunk size.

Not a benchmark: each case says what must show up in the top-k — a partner, a day, and/or a
substring of the chunk text. It runs through whatever retriever the index declares (lexical or
vector), optionally reranked, and exits non-zero on failure. ~1 s, deterministic — the regression
net for refactors, and the replacement for the deleted synthetic eval.

Cases are JSON: `[{"query": …, "partner": …, "day": …, "contains": …}]` — `query` is required,
the rest are conjunctive constraints that must hold for the *same* hit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .index import Index
from .models import Embedder, Reranker
from .search import Hit, search_any

DEFAULT_CASES = [
    {"query": "koden til døren", "partner": "OTHER-630", "day": "2024-09-08",
     "contains": "koden til døren"},
    {"query": "1505#", "partner": "OTHER-630", "contains": "1505#"},
    {"query": "krøllet hår", "partner": "OTHER-001", "day": "2021-11-27"},
    {"query": "ICYE", "partner": "OTHER-197", "contains": "ICYE"},
    {"query": "real madrid", "partner": "OTHER-823", "contains": "Real Madrid"},
    {"query": r"\bfødselsdag\b", "partner": "OTHER-206", "day": "2019-06-05"},
]


@dataclass(slots=True)
class Result:
    query: str
    ok: bool
    rank: int | None
    detail: str = ""
    skipped: bool = False


def _matches(h: "Hit", case: dict) -> bool:
    """A hit may be several merged chunks, so `day` matches any member and `contains` looks at the
    rendered union text."""
    if case.get("partner") and h.partner != case["partner"]:
        return False
    if case.get("day") and case["day"] not in {c.day for c in h.members}:
        return False
    if case.get("contains") and case["contains"] not in h.text:
        return False
    return True


def run(index: Index, cases: list[dict], *, k: int = 5,
        embedder: Embedder | None = None, reranker: Reranker | None = None,
        ) -> tuple[list[Result], dict]:
    # `rank_max` can ask for a hit deeper than k; the pool has to be that deep or the case fails for
    # the wrong reason.
    depth = max([k] + [int(c.get("rank_max") or 0) for c in cases])
    results: list[Result] = []
    for case in cases:
        if case.get("reachable") is False:
            results.append(Result(case["query"], True, None,
                                  case.get("note") or "not reachable by this retriever", skipped=True))
            continue
        pool = search_any(index, case["query"], embedder, k=depth, reranker=reranker)
        rank = next((i + 1 for i, h in enumerate(pool) if _matches(h, case)), None)
        limit = int(case.get("rank_max") or k)
        if rank is not None and rank > limit:
            detail = f"found at #{rank}, allowed #{limit}"
            rank = None
        detail = ""
        if rank is None:
            detail = ("no hits at all" if not pool else
                      "no hit in top-%d (%s)" % (k, ", ".join(
                          f"#{i + 1} {h.partner}/{h.chunk.day}" for i, h in enumerate(pool[:3]))))
        results.append(Result(case["query"], rank is not None, rank, detail))
    ran = [r for r in results if not r.skipped]
    summary = {"n": len(ran), "passed": sum(r.ok for r in ran),
               "top1": sum(1 for r in ran if r.rank == 1),
               "skipped": sum(1 for r in results if r.skipped),
               "kind": index.kind, "rerank": bool(reranker)}
    return results, summary


def derive(index: Index, cases: list[dict], *, embedder: Embedder | None = None,
           reranker: Reranker | None = None, headroom: int = 1) -> list[dict]:
    """Fill `partner` and `rank_max` from the index, so a case cannot go stale against a rebuild.

    The holder is the chunk whose text contains the string; the rank is where this query currently
    reaches that text. `rank_max` is that rank plus one, i.e. a case is allowed to move *up* but not
    to slide silently to the edge of the window.
    """
    out: list[dict] = []
    for case in cases:
        needle = case.get("contains")
        if not needle:
            out.append(case)
            continue
        holder = next((c for c in index.chunks if needle in c.text), None)
        if holder is None:
            out.append({**case, "reachable": False})
            continue
        pool = search_any(index, case["query"], embedder, k=50, reranker=reranker)
        rank = next((i + 1 for i, h in enumerate(pool) if needle in h.text), None)
        out.append({**case, "partner": holder.partner,
                    **({"rank_max": min(50, rank + headroom)} if rank else {"reachable": False})})
    return out


def load_cases(path: str | Path | None) -> list[dict]:
    if path is None:
        return list(DEFAULT_CASES)
    return json.loads(Path(path).read_text())
