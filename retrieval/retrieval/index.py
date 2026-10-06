"""Persist chunks + vectors: chunks.jsonl · vectors.npy · manifest.json."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunk.base import Chunk


@dataclass(slots=True)
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray
    manifest: dict


def save(root: str | Path, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    np.save(root / "vectors.npy", vectors.astype(np.float32))
    manifest = {**manifest, "n_chunks": len(chunks), "dim": int(vectors.shape[1]) if vectors.size else 0,
                "written_at": datetime.now(timezone.utc).isoformat()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def load(root: str | Path) -> Index:
    root = Path(root)
    chunks = [Chunk(**json.loads(line)) for line in (root / "chunks.jsonl").read_text().splitlines()]
    vectors = np.load(root / "vectors.npy")
    manifest = json.loads((root / "manifest.json").read_text())
    return Index(chunks, vectors, manifest)
