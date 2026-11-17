"""Merge the parts of a conversation, on content rather than names.

The donor tool merges two reads of one conversation by the participant signature — the set of real
names — because that is what the export gives it. Here the names are gone, so the join has to be made
from something that survives masking: the messages themselves. Two reads of one conversation overlap
in `(timestamp, text)` pairs; two different conversations do not, or do so by accident.

That is the whole trick, and it is why a merge bug is now a rerun: a `--keep-all` donation carries its
parts in `_parts/`, and this can rejoin them under a different rule without asking anybody for
anything. The signature is deliberately *content*, never identity:

    key(message) = ts_ms, sha1(text)

and two threads are one conversation when the Jaccard overlap of their key sets is at least
`MIN_OVERLAP`, on the same platform. 0.5 is not a tuned number: a genuine second read of one
conversation shares nearly everything, a different conversation shares nothing but a coincidence.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

MIN_OVERLAP = 0.5


def message_key(message: dict) -> tuple[int, str]:
    """Identity of one message that survives masking: when it was sent, and what it said."""
    text = message.get("text") or ""
    return (int(message.get("ts_ms") or 0), hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])


def signature(payload: dict) -> frozenset[tuple[int, str]]:
    keys = {message_key(m) for m in payload.get("messages") or []}
    keys.discard((0, hashlib.sha1(b"").hexdigest()[:16]))
    return frozenset(keys)


def overlaps(a: frozenset, b: frozenset) -> float:
    """Jaccard. Cheap enough at this scale (a thread is thousands of keys, not millions)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def merge_payloads(payloads: Iterable[tuple[str, dict]]) -> list[dict]:
    """Fold `(source, payload)` pairs into conversations.

    `source` is kept in `sources` so the result knows which reads it came from, exactly like a thread
    the donor tool merged — the difference is only what the join was made of.
    """
    groups: list[dict] = []
    for source, payload in payloads:
        keys = signature(payload)
        for group in groups:
            if group["platform"] != payload.get("platform", group["platform"]):
                continue
            if overlaps(keys, group["keys"]) >= MIN_OVERLAP:
                _absorb(group, source, payload)
                break
        else:
            groups.append(
                {
                    "platform": payload.get("platform"),
                    "participants": list(payload.get("participants") or []),
                    "sources": [source],
                    "messages": list(payload.get("messages") or []),
                    "keys": keys,
                }
            )
    for group in groups:
        group["messages"].sort(key=lambda m: int(m.get("ts_ms") or 0))
        # A message the two reads both hold is one message. The text is the same by construction (it is
        # in the key), so the surviving copy needs no reconciliation.
        seen: set[tuple[int, str]] = set()
        kept = []
        for message in group["messages"]:
            key = message_key(message)
            if key in seen:
                for extra in ("as_prompt",):
                    message.pop(extra, None)
                continue
            seen.add(key)
            kept.append(message)
        group["messages"] = kept
        group["n_messages"] = len(kept)
    for group in groups:
        group.pop("keys", None)
    return groups


def _absorb(group: dict, source: str, payload: dict) -> None:
    group["sources"] = sorted({*group["sources"], source})
    group["messages"].extend(payload.get("messages") or [])
    group["keys"] = group["keys"] | signature(payload)
    group["participants"] = sorted(set(group["participants"]) | set(payload.get("participants") or []))
