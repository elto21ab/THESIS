"""Smoke test: hand-written (query → expected) pairs over a real index.

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
    results: list[Result] = []
    for case in cases:
        pool = search_any(index, case["query"], embedder, k=k, reranker=reranker)
        rank = next((i + 1 for i, h in enumerate(pool) if _matches(h, case)), None)
        detail = ""
        if rank is None:
            detail = ("no hits at all" if not pool else
                      "no hit in top-%d (%s)" % (k, ", ".join(
                          f"#{i + 1} {h.partner}/{h.chunk.day}" for i, h in enumerate(pool[:3]))))
        results.append(Result(case["query"], rank is not None, rank, detail))
    summary = {"n": len(results), "passed": sum(r.ok for r in results),
               "top1": sum(1 for r in results if r.rank == 1), "kind": index.kind,
               "rerank": bool(reranker)}
    return results, summary


def load_cases(path: str | Path | None) -> list[dict]:
    if path is None:
        return list(DEFAULT_CASES)
    return json.loads(Path(path).read_text())
