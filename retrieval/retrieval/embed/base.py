"""Embedder protocol + a deterministic stub so the pipeline runs before the model lands."""
from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    dim: int

    def count(self, text: str) -> int: ...
    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class StubEmbedder:
    """Word-hash bag-of-words. No model, no download — smoke-tests chunk/io/index/retrieve."""

    dim = 256

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for word in t.lower().split():
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big")
                out[r, h % self.dim] += 1.0
        return normalize(out)
