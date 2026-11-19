"""Merge the parts of a conversation, on content rather than names.

The donor tool merges two reads of one conversation by the participant signature — the set of real
names — because that is what the export gives it. Here the names are gone, so the join has to be made
from something that survives masking: the messages themselves. Two reads of one conversation overlap
in `(timestamp, text)` pairs; two different conversations do not, or do so by accident.

That is the whole trick, and it is why a merge bug is now a rerun: a `--keep-all` donation carries its
parts in `_parts/`, and this can rejoin them under a different rule without asking anybody for
anything. The signature is deliberately *content*, never identity:

    key(message) = ts_ms, sha1(text)

and two threads are one conversation when **most of the smaller one is inside the larger** — the
overlap coefficient, `|A ∩ B| / min(|A|, |B|)`, at least `MIN_OVERLAP`, on the same platform.

Not Jaccard, and the difference is the whole point: one export often covers *part* of a conversation
(the dump before a device change, the cutover after it), so the shared half is a subset, not a twin.
Jaccard punishes the size difference and refused to merge exactly the pairs this exists to merge — 9
shared messages out of 20-and-14 is 0.36 by Jaccard and 0.64 by containment.

`MIN_SHARED` keeps the metric honest on tiny payloads: two two-message threads sharing one message
score 0.5 by containment without being evidence of anything.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

MIN_OVERLAP = 0.5
MIN_SHARED = 5


def identity_signature(participants: list[str]) -> str:
    """FNV-1a 64 over the lowercased, sorted names — the same function as `identitySignature` in
    `thesis-deid/src/core/standard/format.ts`. Two reads of one conversation share it even when the
    reads are disjoint in time, which is the case content overlap cannot see."""
    names = sorted({name.strip().lower() for name in participants if name.strip()})
    hash_ = 0xCBF29CE484222325
    prime = 0x100000001B3
    mask = 0xFFFFFFFFFFFFFFFF
    for byte in "\n".join(names).encode("utf-8"):
        hash_ ^= byte
        hash_ = (hash_ * prime) & mask
    return f"{hash_:016x}"


def message_key(message: dict) -> tuple[int, str]:
    """Identity of one message that survives masking: when it was sent, and what it said."""
    text = message.get("text") or ""
    return (int(message.get("ts_ms") or 0), hashlib.sha1(text.encode("utf-8")).hexdigest()[:16])


def signature(payload: dict) -> frozenset[tuple[int, str]]:
    keys = {message_key(m) for m in payload.get("messages") or []}
    keys.discard((0, hashlib.sha1(b"").hexdigest()[:16]))
    return frozenset(keys)


def overlaps(a: frozenset, b: frozenset) -> float:
    """How much of the smaller conversation is present in the larger (the overlap coefficient).

    Cheap at this scale: a thread is thousands of keys, and a platform has hundreds of threads, so the
    comparison is set arithmetic over a few million hashes at worst.
    """
    if not a or not b:
        return 0.0
    shared = len(a & b)
    if shared < MIN_SHARED:
        return 0.0
    return shared / min(len(a), len(b))


def merge_payloads(payloads: Iterable[tuple[str, dict]]) -> list[dict]:
    """Fold `(source, payload)` pairs into conversations.

    `source` is kept in `sources` so the result knows which reads it came from, exactly like a thread
    the donor tool merged — the difference is only what the join was made of.
    """
    groups: list[dict] = []
    for source, payload in payloads:
        keys = signature(payload)
        # The fingerprint first: it is exact, and it is the only evidence left when a device switch
        # left the two reads with no messages in common. Content overlap is the fallback for a bundle
        # written before the field existed.
        # Only an explicit fingerprint counts. Deriving one from `participants` would hash labels: every
        # archived thread puts the donor at `S` and the partner at `P1`, so the derived signature is the
        # same for all of them and a re-merge folds the platform into a single conversation.
        sig = payload.get("identity_sig")
        for group in groups:
            if group["platform"] != payload.get("platform", group["platform"]):
                continue
            same_identity = bool(sig) and sig == group["sig"]
            if same_identity or overlaps(keys, group["keys"]) >= MIN_OVERLAP:
                _absorb(group, source, payload)
                break
        else:
            groups.append(
                {
                    "platform": payload.get("platform"),
                    "sig": sig,
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
        group.pop("sig", None)
    return groups


def _absorb(group: dict, source: str, payload: dict) -> None:
    group["sources"] = sorted({*group["sources"], source})
    group["messages"].extend(payload.get("messages") or [])
    group["keys"] = group["keys"] | signature(payload)
    group["participants"] = sorted(set(group["participants"]) | set(payload.get("participants") or []))
