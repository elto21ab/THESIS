from __future__ import annotations

import hashlib

RENDER_VERSION = "v1"


def chunk_id(platform: str, partner: str, i0: int, i1: int, ts0: int, ts1: int) -> str:
    """Stable across re-embed and re-run; changes only if the span or render changes."""
    key = f"{RENDER_VERSION}|{platform}|{partner}|{i0}|{i1}|{ts0}|{ts1}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
