"""CLI: build an index from a bundle, or query one.

    uv run python -m retrieval.cli build --bundle DIR --out DIR [--chunker fixed|day] [--embedder stub|jina]
    uv run python -m retrieval.cli query --index DIR "some text" [-k 8]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .chunk import CHUNKERS
from .chunk.base import ChunkCfg
from .embed import get_embedder
from .index import load, save
from .io import load_bundle
from .retrieve import retrieve

DEFAULT_BUNDLE = "/Users/e/Downloads/donation-2026-09-19"


def _make_embedder(a: argparse.Namespace):
    if a.embedder == "jina":
        return get_embedder("jina", dim=a.dim, batch=a.batch, max_length=a.max_length,
                            dtype=a.dtype or None)
    if a.embedder == "openai":
        return get_embedder("openai", model=a.model, base_url=a.base_url, dim=a.dim,
                            batch=a.batch, task_prompt=not a.no_task_prompt)
    return get_embedder(a.embedder)


def _add_embed_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedder", default="stub", help="stub | jina (MPS) | openai (llama.cpp/vLLM)")
    p.add_argument("--dim", type=int, default=1024)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max-length", type=int, default=2048, dest="max_length")
    p.add_argument("--dtype", help="jina only: float16 | bfloat16 | float32")
    p.add_argument("--base-url", default="http://127.0.0.1:8091/v1", dest="base_url")
    p.add_argument("--model", help="openai only, e.g. bge-m3 or the served model id")
    p.add_argument("--no-task-prompt", action="store_true", dest="no_task_prompt",
                   help="openai: skip the Query:/Document: prefix (non-Jina servers)")


def cmd_build(a: argparse.Namespace) -> None:
    cfg = ChunkCfg(target=a.target, min=a.min, max=a.max, overlap=a.overlap,
                   tz=a.tz, day_start_hour=a.day_start, both_people=not a.no_both)
    embedder = _make_embedder(a)
    threads, rejects, _ = load_bundle(a.bundle, max_senders=a.max_senders)
    if a.platform:
        threads = [t for t in threads if t.platform == a.platform]
    if a.partner:
        threads = [t for t in threads if t.partner == a.partner]
    if a.limit:
        threads = threads[:a.limit]
    if a.sample:
        threads = sorted(threads, key=lambda t: len(t.messages))[:a.sample]

    chunker = CHUNKERS[a.chunker]
    chunks = []
    for t in threads:
        chunks.extend(chunker(t, cfg, embedder.count))

    vectors = embedder.embed([c.text for c in chunks])
    manifest = {
        "bundle": str(a.bundle), "chunker": a.chunker, "embedder": a.embedder,
        "cfg": cfg.__dict__ if hasattr(cfg, "__dict__") else {
            k: getattr(cfg, k) for k in cfg.__slots__},
        "threads": len(threads), "rejected": len(rejects),
    }
    save(a.out, chunks, vectors, manifest)

    tok = [c.n_tokens for c in chunks]
    print(f"threads kept : {len(threads)}")
    print(f"rejected     : {len(rejects)}  {dict(Counter(r.reason for r in rejects))}")
    print(f"chunks       : {len(chunks)}")
    if tok:
        tok.sort()
        print(f"tokens/chunk : min {tok[0]} · p50 {tok[len(tok)//2]} · max {tok[-1]} "
              f"· mean {sum(tok)//len(tok)}")
    print(f"written      : {Path(a.out).resolve()}")
    for r in rejects[:10]:
        print(f"  reject {r.reason:11} {r.path}  {r.detail}")
    if len(rejects) > 10:
        print(f"  … +{len(rejects) - 10} more")


def cmd_query(a: argparse.Namespace) -> None:
    idx = load(a.index)
    embedder = _make_embedder(a)
    hits = retrieve(idx, a.text, embedder, k=a.k, platform=a.platform, partner=a.partner)
    for rank, h in enumerate(hits, 1):
        c = h.chunk
        head = (f"#{rank} {h.score:+.3f}  {c.platform}/{c.partner} seq{c.seq} "
                f"msgs[{c.i0}:{c.i1}] {c.n_tokens}tok {c.day}")
        print(head)
        print("  " + c.text[:400].replace("\n", "\n  "))
        print()


def cmd_eval(a: argparse.Namespace) -> None:
    from .eval import evaluate, synthetic
    from .rerank import HttpReranker

    idx = load(a.index)
    embedder = _make_embedder(a)
    qs = synthetic(idx.chunks, n=a.n, seed=a.seed)
    if a.platform:
        qs = [q for q in qs if q.platform == a.platform]
    reranker = None
    if a.rerank:
        reranker = HttpReranker(base_url=a.rerank_url, model=a.rerank_model, path=a.rerank_path)
    out = evaluate(idx, embedder, qs, k=a.k, candidates=a.candidates, reranker=reranker)
    tag = f"rerank({a.rerank_model})" if a.rerank else "vector"
    print(f"index={a.index}  {tag}  candidates={a.candidates if a.rerank else '-'}  {out}")


def main() -> None:
    p = argparse.ArgumentParser(prog="retrieval")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="chunk + embed a bundle into an index")
    b.add_argument("--bundle", default=DEFAULT_BUNDLE)
    b.add_argument("--out", required=True)
    b.add_argument("--chunker", choices=sorted(CHUNKERS), default="day")
    _add_embed_args(b)
    b.add_argument("--target", type=int, default=512)
    b.add_argument("--min", type=int, default=256)
    b.add_argument("--max", type=int, default=1024)
    b.add_argument("--overlap", type=float, default=0.40)
    b.add_argument("--tz", default="Europe/Copenhagen")
    b.add_argument("--day-start", type=int, default=5)
    b.add_argument("--no-both", action="store_true", help="disable both-people soft extend")
    b.add_argument("--max-senders", type=int, default=2)
    b.add_argument("--platform", help="only this platform (fb|ig|wa)")
    b.add_argument("--partner", help="only this partner, e.g. OTHER-052")
    b.add_argument("--limit", type=int, default=0, help="only first N threads (debug)")
    b.add_argument("--sample", type=int, default=0, help="N smallest threads (bounded demo)")
    b.set_defaults(func=cmd_build)

    q = sub.add_parser("query", help="query an index")
    q.add_argument("--index", required=True)
    q.add_argument("text")
    q.add_argument("-k", type=int, default=8)
    _add_embed_args(q)
    q.add_argument("--platform")
    q.add_argument("--partner")
    q.set_defaults(func=cmd_query)

    e = sub.add_parser("eval", help="synthetic-query evaluation of an index")
    e.add_argument("--index", required=True)
    _add_embed_args(e)
    e.add_argument("--n", type=int, default=100, help="number of synthetic queries")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("-k", type=int, default=10)
    e.add_argument("--candidates", type=int, default=50, help="vector pool fed to the reranker")
    e.add_argument("--platform")
    e.add_argument("--rerank", action="store_true")
    e.add_argument("--rerank-url", default="http://127.0.0.1:8092", dest="rerank_url")
    e.add_argument("--rerank-model", default="jina-reranker-v3.5", dest="rerank_model")
    e.add_argument("--rerank-path", default="/reranking", dest="rerank_path")
    e.set_defaults(func=cmd_eval)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
