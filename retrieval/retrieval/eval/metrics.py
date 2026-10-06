"""Metrics over a list of target ranks (1-based; None = not retrieved)."""
from __future__ import annotations


def hit_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r and r <= k) / len(ranks) if ranks else 0.0


def mrr(ranks: list[int | None]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def summarize(ranks: list[int | None]) -> dict:
    return {
        "n": len(ranks),
        "hit@1": round(hit_at(ranks, 1), 3),
        "hit@5": round(hit_at(ranks, 5), 3),
        "hit@10": round(hit_at(ranks, 10), 3),
        "mrr": round(mrr(ranks), 3),
    }
