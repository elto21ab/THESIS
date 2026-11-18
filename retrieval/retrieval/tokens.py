"""How many tokens is this text, really.

The whole pipeline used `len(text) // 4` — the char-per-token rule of thumb from English prose. It is
wrong for this corpus, and wrong in a direction that hides overflow: measured on 100 chunks of the
real thing, **2.21 chars/token** for Qwen3-Embedding-0.6B and **2.31** for Qwen3.5-0.8B. That is a
1.8× underestimate, which turns a "40k token budget" into 72k and an 8k context into an overflow.

Two honest ways to count, and when each is right:

* **`exact()`** — `llama-tokenize` with the model's own gguf. Free (a binary, no model load), ~0.2 s
  for 100k chars, and exact. Use it at build time on a sample, or on a single string you are about to
  send.
* **`calibrate()`** — one batched `exact()` over a sample of the corpus, giving chars-per-token for
  *this* model and *this* language mix. Store it in the index manifest and multiply. Costs one second
  per build and is right to within a few percent for the rest of the corpus, which is all a budget
  needs.

The ratio is per model because tokenizers differ: 0.6B embedder 2.21, 0.8B chat 2.31, and a 27B model
will differ again. A hard-coded 4 is the one choice that is never right.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

#: Fallback when no gguf is reachable. Measured, not guessed: 2.3 chars/token on Danish chat text.
DEFAULT_RATIO = 2.3


def available() -> bool:
    return shutil.which("llama-tokenize") is not None


def exact(texts: Sequence[str], model: str | Path) -> int | None:
    """Token count with the model's own vocabulary, or None if the tool or the gguf is missing."""
    if not available():
        return None
    gguf = Path(model)
    if not gguf.exists():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("\n".join(t.replace("\n", " ") for t in texts))
        path = fh.name
    try:
        out = subprocess.run(["llama-tokenize", "-m", str(gguf), "-f", path, "--ids"],
                             capture_output=True, text=True, timeout=600)
        if out.returncode != 0:
            return None
        return sum(1 for chunk in out.stdout.replace("[", " ").replace("]", " ").split(",")
                   for tok in chunk.split() if tok.lstrip("-").isdigit())
    finally:
        Path(path).unlink(missing_ok=True)


def calibrate(texts: Iterable[str], model: str | Path, *, sample: int = 64) -> float:
    """Chars per token for this model on this corpus. Falls back to `DEFAULT_RATIO`."""
    picked = [t for t in list(texts)[:sample] if t]
    if not picked:
        return DEFAULT_RATIO
    n = exact(picked, model)
    if not n:
        return DEFAULT_RATIO
    chars = sum(len(t) for t in picked)
    return max(1.0, chars / n)


def count(text: str, ratio: float = DEFAULT_RATIO) -> int:
    return max(1, int(len(text) / ratio))


def load_ratio(index_manifest: dict) -> float:
    """The ratio a build recorded. Older indexes have none, and 4 was their implicit assumption."""
    return float(index_manifest.get("chars_per_token") or 3.0)


if __name__ == "__main__":  # calibration of one corpus by hand
    import sys

    paths = sys.argv[2:]
    texts = []
    for p in paths:
        texts.extend(json.loads(l)["text"] for l in open(p))
    print(calibrate(texts, sys.argv[1]))
