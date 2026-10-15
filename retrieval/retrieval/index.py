"""On-disk index: `chunks.jsonl` + (`vectors.npy` | `lexical.bin`+`offsets.npy`) + `manifest.json`.

Two kinds, one metadata record:
  vector  — `vectors.npy`, float32 `[n × dim]`, L2-normalised; retrieval is one matmul.
  lexical — `lexical.bin`, the chunk texts concatenated (`\\n\\0\\n`-separated) with `offsets.npy`
            giving each chunk's **character** start; retrieval is one sequential regex scan.
`chunks.jsonl` is the record for both: id, partner, day, span, tokens, header, text — everything a
retriever, `get`, the token budget and the filters need. The manifest carries the facts a query
must not guess (kind, embedder identity, donor, bundle) and the config intent captured at build.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .chunk import Chunk

SEP = "\n\x00\n"
BLOB = "lexical.bin"
OFFSETS = "offsets.npy"


@dataclass(slots=True)
class Index:
    chunks: list[Chunk]
    vectors: np.ndarray | None
    manifest: dict
    root: Path | None = None

    @property
    def kind(self) -> str:
        return self.manifest.get("kind", "vector")


def save(root: str | Path, chunks: list[Chunk], vectors: np.ndarray | None, manifest: dict,
         *, blob: bool = False) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "chunks.jsonl").open("w") as f:
        for c in chunks:
            f.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    if vectors is not None:
        np.save(root / "vectors.npy", vectors.astype(np.float32))
    if blob:
        write_blob(root, chunks)
    manifest = {**manifest, "n_chunks": len(chunks),
                "dim": int(vectors.shape[1]) if vectors is not None and vectors.size else 0,
                "written_at": datetime.now(timezone.utc).isoformat()}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def load(root: str | Path) -> Index:
    root = Path(root)
    chunks = [Chunk(**json.loads(line)) for line in (root / "chunks.jsonl").read_text().splitlines()]
    vp = root / "vectors.npy"
    return Index(chunks, np.load(vp) if vp.exists() else None,
                 json.loads((root / "manifest.json").read_text()), root)


# ── lexical blob ──────────────────────────────────────────────────────────────
def write_blob(root: str | Path, chunks: list[Chunk]) -> dict:
    """Write lexical.bin + offsets.npy (character offsets). Returns {n_chunks, bytes}."""
    root = Path(root)
    offs = np.zeros(len(chunks) + 1, dtype=np.int64)
    with (root / BLOB).open("wb") as f:
        pos = 0
        for i, c in enumerate(chunks):
            offs[i] = pos
            f.write((c.text + SEP).encode("utf-8"))
            pos += len(c.text) + len(SEP)
        offs[len(chunks)] = pos
    np.save(root / OFFSETS, offs)
    return {"n_chunks": len(chunks), "bytes": (root / BLOB).stat().st_size}


def read_blob(root: str | Path) -> tuple[str, np.ndarray] | tuple[None, None]:
    """The whole corpus text + per-chunk character offsets. One decode (~10 ms at 11 MB).

    Read as `str` (not bytes) so case-folding works for Danish and match offsets line up with
    `offsets.npy` without a byte↔char conversion.
    """
    root = Path(root)
    if not (root / BLOB).exists():
        return None, None
    return (root / BLOB).read_text(encoding="utf-8"), np.load(root / OFFSETS)
