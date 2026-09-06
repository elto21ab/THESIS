"""Merge + dedupe.

Merge is per-platform (FB DYI + E2EE dump join on participant signature; other
platforms are single-source). Same person across platforms stays separate threads —
cross-platform identity is a later, explicit concern.

Dedupe is content-level, after merge: two messages are duplicates when they share
(ts, sender, folded-text). That kills the FB DYI/E2EE overlap (~every e2ee thread
appears in both, DYI up to backup date).
"""
from __future__ import annotations

from collections import Counter

from format import Thread, thread_sig


def merge_threads(threads: list[Thread]) -> list[Thread]:
    """Join threads whose participant signature matches; concatenate messages."""
    by_sig: dict[str, Thread] = {}
    for t in threads:
        sig = thread_sig(t.participants)
        if not sig:
            continue
        if sig in by_sig:
            target = by_sig[sig]
            target.messages.extend(t.messages)
            target.sources = sorted(set(target.sources) | set(t.sources))
            target.participants = sorted(set(target.participants) | set(t.participants))
        else:
            by_sig[sig] = t
    return list(by_sig.values())


def dedupe_thread(t: Thread) -> int:
    """Drop exact content-duplicates. Returns number dropped."""
    seen: set[str] = set()
    kept: list = []
    dropped = 0
    for m in t.messages:
        if m.system:
            kept.append(m)
            continue
        k = m.dedupe_key()
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        kept.append(m)
    t.messages = kept
    t.finalise()
    return dropped


def dedupe_threads(threads: list[Thread]) -> tuple[list[Thread], int]:
    total = 0
    for t in threads:
        total += dedupe_thread(t)
    return threads, total


def pick_owner(threads: list[Thread]) -> str:
    """Most frequent participant name across all merged threads (raw)."""
    c: Counter[str] = Counter()
    for t in threads:
        for p in t.participants:
            c[p.strip().lower()] += 1
    return c.most_common(1)[0][0] if c else ""


def finalise(threads: list[Thread], prefix: str, owner: str = "") -> list[Thread]:
    """Assign ids (fb001…) and set partner = the non-owner participant."""
    owner_l = owner.strip().lower()
    for t in threads:
        others = [p for p in t.participants if p.strip().lower() != owner_l]
        t.partner = others[0] if others else (t.participants[0] if t.participants else "")
    threads.sort(key=lambda t: (-len(t.messages), t.partner.lower()))
    for i, t in enumerate(threads, 1):
        t.id = f"{prefix}{i:03d}"
    return threads


def filter_threads(
    threads: list[Thread],
    min_msgs: int = 20,
    # TODO heuristic backlog: revisit min_active_days + min_days later.
    # min_active_days: int = 2,   # distinct calendar days w/ >=1 msg
    # min_days: int = 7,          # first→last span (was dropped for single bursts)
    owner: str = "",
) -> tuple[list[Thread], dict]:
    """Drop long-tail conversations with minimal signal.

    Active filter: a thread survives when it has >= min_msgs text messages.
    (Heuristics for active-days / span are parked — see TODO above.)
    """
    kept: list[Thread] = []
    dropped = {"min_msgs": 0, "total": 0}
    for t in threads:
        text = [m for m in t.messages if not m.system]
        n = len(text)
        if n < min_msgs:
            dropped["min_msgs"] += 1
        else:
            kept.append(t)
    dropped["total"] = dropped["min_msgs"]
    return kept, dropped