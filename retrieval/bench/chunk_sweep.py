"""Chunker shape sweep — no embeddings, just structure.

Sweep target size x overlap x chunker and report: chunk count, tokens/chunk percentiles,
share below `min`, and the duplication factor (sum of chunk tokens / corpus tokens), which is
what overlap and day-merging cost. Embedding each config is slow, so this narrows the field first.
"""
from retrieval.chunk import CHUNKERS
from retrieval.chunk.base import ChunkCfg
from retrieval.embed.base import StubEmbedder
from retrieval.io import load_bundle

BUNDLE = "/Users/e/Downloads/donation-2026-09-19"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def main() -> None:
    threads, _, _ = load_bundle(BUNDLE)
    e = StubEmbedder()
    corpus = sum(e.count("\n".join(f"{m.sender} {m.text}" for m in t.messages)) for t in threads)
    print(f"threads={len(threads)}  corpus~{corpus} tok")
    print(f"{'chunker':6} {'target':>6} {'ov':>4} {'chunks':>7} {'tok p10/50/90':>16} {'<min%':>6} {'dup':>5}")

    for name in ("fixed", "day"):
        for target in (256, 512, 1024):
            for ov in (0.0, 0.2, 0.4):
                cfg = ChunkCfg(target=target, min=target // 2, max=target * 2, overlap=ov)
                chunks = []
                for t in threads:
                    chunks.extend(CHUNKERS[name](t, cfg, e.count))
                tok = [c.n_tokens for c in chunks]
                below = sum(1 for x in tok if x < cfg.min) / len(tok)
                dup = sum(tok) / corpus
                print(f"{name:6} {target:6} {ov:4.1f} {len(chunks):7} "
                      f"{pct(tok,.1):5}/{pct(tok,.5):5}/{pct(tok,.9):5} {below*100:5.0f}% {dup:5.2f}")


if __name__ == "__main__":
    main()
