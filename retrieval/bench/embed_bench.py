"""Quick embedder throughput probe. Bounded: a handful of items per config."""
import time

import torch

from retrieval.embed.local import JinaLocalEmbedder


def bench(device: str, n: int = 8, words: int = 200) -> None:
    e = JinaLocalEmbedder(dim=1024, max_length=2048, device=device, batch=n)
    p = next(e.model.parameters())
    line = "[2024-01-19 12:07:30] SUBJECT: " + ("hej " * words)
    texts = [line] * n
    e.embed(texts[:1])  # warmup (compile/first alloc)
    t = time.time()
    e.embed(texts)
    dt = time.time() - t
    print(f"{device:5} dtype={str(p.dtype):8} n={n:2} ~{words*2:5}tok  "
          f"{dt:6.2f}s  {dt/n*1000:7.0f} ms/item", flush=True)


if __name__ == "__main__":
    for dev in ("mps", "cpu"):
        for n in (1, 4, 16):
            try:
                bench(dev, n=n)
            except Exception as exc:  # noqa: BLE001
                print(f"{dev} n={n} FAILED: {type(exc).__name__}: {exc}", flush=True)
    if torch.backends.mps.is_available():
        print("mps available")
