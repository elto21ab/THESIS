"""Throughput probe: dtype x batch at ~1000-token items. Bounded, few configs."""
import time

from retrieval.embed.local import JinaLocalEmbedder

WORDS = 500  # ~1000 tokens


def run(tag, **kw):
    e = JinaLocalEmbedder(dim=1024, max_length=1024, device="mps", **kw)
    texts = ["[2024-01-19 12:07:30] SUBJECT: " + ("hej " * WORDS)] * 32
    e.embed(texts[:1])  # warmup
    t = time.time()
    e.embed(texts)
    dt = time.time() - t
    print(f"{tag:28} {dt:6.2f}s  {dt/32*1000:6.0f} ms/item", flush=True)


for tag, kw in [
    ("bf16 batch=16", dict(batch=16)),
    ("bf16 batch=32", dict(batch=32)),
    ("fp16 batch=32", dict(batch=32, dtype="float16")),
]:
    try:
        run(tag, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"{tag:28} FAILED {type(exc).__name__}: {exc}", flush=True)
