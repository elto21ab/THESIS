"""CLI: build an index, or drive retrieval as a tool. Settings come from `config.yaml`.

    uv run python -B -m retrieval build --bundle DIR --out DIR [--kind vector|lexical] [--donor-id ID]
    uv run python -B -m retrieval push  [--probe S ... | --probes-file F] [-k N] [--budget N]
    uv run python -B -m retrieval pull search "text" [-k N] [--since D] [--until D] [--platform P]
    uv run python -B -m retrieval pull get  <id> [--window N]
    uv run python -B -m retrieval smoke [-k N] [--cases cases.json]

JSON is the contract (default); `--pretty` renders for humans. `$RETRIEVAL_INDEX` supplies the
index (one index == one donor) and `$RETRIEVAL_DONOR` asserts the donor id. Two config levels only:
a CLI flag overrides `config.yaml` (`--config config.server.yaml` switches environment).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml

from .bundle import load_bundle
from .chunk import CHUNKERS, ChunkCfg
from .index import Index, load, save
from .models import HttpReranker, get_embedder
from .ops import chunk_payload, get, hit_payload, hits_payload, pack
from .search import search_any
from .loop import run as run_loop
from .smoke import load_cases, run as run_smoke

DEFAULT_BUNDLE = "/Users/e/Downloads/donation-2026-09-19"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
# Every key the code reads; the file must supply them all (no hidden defaults, no profiles).
REQUIRED = ("chunk.name chunk.target chunk.min chunk.max chunk.overlap chunk.tz chunk.day_start_hour"
            " chunk.both_people bundle.max_senders embed.backend embed.model embed.base_url embed.batch"
            " embed.prompt embed.max_length rerank.enabled rerank.model rerank.base_url rerank.path"
            " rerank.prompt rerank.pool llm.model llm.base_url llm.temperature llm.max_turns llm.max_calls retrieve.k retrieve.budget retrieve.min_threads retrieve.max_chars"
            " retrieve.max_region_chunks").split()


def _cfg(a: argparse.Namespace) -> dict:
    """Config precedence is exactly two levels: a CLI flag overrides `config.yaml`. Switching
    environments means pointing at a different file: `--config config.server.yaml`."""
    path = Path(a.config) if getattr(a, "config", None) else DEFAULT_CONFIG
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in REQUIRED if _dig(cfg, k) is None]
    if missing:
        raise SystemExit(f"{path}: missing config keys: {', '.join(missing)}")
    return cfg


def _dig(cfg: dict, dotted: str):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


# ── plumbing ──────────────────────────────────────────────────────────────────
def _index_path(a: argparse.Namespace) -> str:
    p = a.index or os.environ.get("RETRIEVAL_INDEX")
    if not p:
        raise SystemExit("no index: pass --index or set RETRIEVAL_INDEX")
    return p


def _donor(idx: Index, a: argparse.Namespace) -> str | None:
    """One index == one donor; abort when pointed at the wrong one."""
    got = idx.manifest.get("donor_id")
    want = a.expect_donor or os.environ.get("RETRIEVAL_DONOR")
    if want and got != want:
        raise SystemExit(f"donor mismatch: index donor_id={got!r}, expected {want!r}")
    return got


def _embedder(idx: Index | None, cfg: dict | None = None):
    """Query: read the manifest (the fact). Warn if the config intent disagrees — the manifest
    always wins, because the vectors were built with it and a blocking error would make old
    indexes unqueryable. Build: read the config (the intent)."""
    if idx is None:
        return get_embedder(cfg["embed"]["backend"], model=cfg["embed"]["model"],
                            base_url=cfg["embed"]["base_url"], batch=cfg["embed"]["batch"],
                            prompt=cfg["embed"]["prompt"], max_length=cfg["embed"]["max_length"])
    if idx.kind == "lexical":
        return None
    c = idx.manifest.get("embedder_cfg")
    if not c:
        raise SystemExit("index manifest lacks `embedder_cfg`; rebuild the index")
    if cfg and cfg["embed"]["model"] != c.get("model"):
        print(f"WARN config says embed.model={cfg['embed']['model']!r} but the index was built "
              f"with {c.get('model')!r} — using the index (rebuild to change)", file=sys.stderr)
    return get_embedder(c.get("backend", "openai"), model=c["model"], base_url=c["base_url"],
                        batch=c.get("batch", 32), prompt=c.get("prompt", "none"),
                        max_length=c.get("max_length"))


def _reranker(cfg: dict, *, enabled: bool | None = None) -> HttpReranker | None:
    r = cfg["rerank"]
    on = r["enabled"] if enabled is None else enabled
    return HttpReranker(model=r["model"], base_url=r["base_url"], path=r["path"],
                        prompt=r["prompt"]) if on else None


def _emit(obj: dict, a: argparse.Namespace, pretty: str | None = None) -> None:
    if a.pretty and pretty:
        print(pretty, end="" if pretty.endswith("\n") else "\n")
    else:
        print(json.dumps(obj, ensure_ascii=False))


def _preamble(idx: Index, a: argparse.Namespace) -> str:
    return f"donor={_donor(idx, a)}  kind={idx.kind}"


# ── build ─────────────────────────────────────────────────────────────────────
def cmd_build(a: argparse.Namespace) -> None:
    cfg = _cfg(a)
    kind = a.kind or ("lexical" if cfg["embed"]["backend"] == "lexical" else "vector")
    cc = cfg["chunk"]
    chunk_cfg = ChunkCfg(target=cc["target"], min=cc["min"], max=cc["max"], overlap=cc["overlap"],
                         tz=cc["tz"], day_start_hour=cc["day_start_hour"],
                         both_people=cc["both_people"])
    embedder = None if kind == "lexical" else _embedder(None, cfg)
    threads, rejects, report = load_bundle(a.bundle, max_senders=cfg["bundle"]["max_senders"])
    if a.platform:
        threads = [t for t in threads if t.platform == a.platform]
    if a.partner:
        threads = [t for t in threads if t.partner == a.partner]
    if a.limit:
        threads = threads[:a.limit]
    if a.sample:
        threads = sorted(threads, key=lambda t: len(t.messages))[:a.sample]

    count = embedder.count if embedder else (lambda t: max(1, len(t) // 4))
    chunker = CHUNKERS[cc["name"]]
    chunks: list = []
    for t in threads:
        chunks.extend(chunker(t, chunk_cfg, count))
    vectors = None if kind == "lexical" else embedder.embed([c.text for c in chunks])

    donor_id = a.donor_id or report.get("donor_id") or report.get("subject_id")
    if not donor_id:
        name = Path(a.bundle).name
        donor_id = None if name.startswith("donation-") else name
    if not donor_id:
        print("WARN donor_id unset — pass --donor-id (uploads should name the bundle dir "
              "<donor_id> and/or put `donor_id` in report.json)")

    manifest = {
        "bundle": str(a.bundle), "kind": kind, "chunker": cc["name"], "donor": report.get("donor"),
        "donor_id": donor_id, "bundle_generated_at": report.get("generated_at"),
        "threads": len(threads), "rejected": len(rejects), "cfg": cfg,
        "embedder": None if kind == "lexical" else (cfg["embed"]["model"] if embedder else None),
        "embedder_cfg": None if kind == "lexical" else {
            "backend": cfg["embed"]["backend"], "model": cfg["embed"]["model"],
            "base_url": cfg["embed"]["base_url"], "batch": cfg["embed"]["batch"],
            "prompt": cfg["embed"]["prompt"], "max_length": cfg["embed"]["max_length"]},
    }
    save(a.out, chunks, vectors, manifest, blob=(kind == "lexical"))

    tok = sorted(c.n_tokens for c in chunks)
    print(f"donor        : {donor_id or '?<unset>'}  (bundle {a.bundle})")
    print(f"kind         : {kind}   chunker: {cc['name']}")
    print(f"threads kept : {len(threads)}   rejected: {len(rejects)} "
          f"{dict(Counter(r.reason for r in rejects))}")
    print(f"chunks       : {len(chunks)}")
    if tok:
        print(f"tokens/chunk : min {tok[0]} · p50 {tok[len(tok) // 2]} · max {tok[-1]} "
              f"· mean {sum(tok) // len(tok)}")
    print(f"written      : {Path(a.out).resolve()}")
    for r in rejects[:10]:
        print(f"  reject {r.reason:11} {r.path}  {r.detail}")
    if len(rejects) > 10:
        print(f"  … +{len(rejects) - 10} more")


# ── pull ──────────────────────────────────────────────────────────────────────
def cmd_search(a: argparse.Namespace) -> None:
    idx = load(_index_path(a))
    donor = _donor(idx, a)          # fail fast: wrong index is the one mistake worth catching early
    cfg = _cfg(a)
    hits = search_any(idx, a.text, _embedder(idx, cfg), k=a.k or cfg["retrieve"]["k"],
                      platform=a.platform, partner=a.partner, since=a.since, until=a.until,
                      max_region_chunks=cfg["retrieve"]["max_region_chunks"],
                      reranker=_reranker(cfg, enabled=a.rerank), pool=cfg["rerank"]["pool"])
    max_chars = cfg["retrieve"]["max_chars"]
    if not a.pretty:
        return _emit({"donor": donor, "kind": idx.kind, "query": a.text,
                      "k": a.k or cfg["retrieve"]["k"],
                      "hits": hits_payload(hits, max_chars=max_chars)}, a)
    lex = idx.kind == "lexical"
    body = _preamble(idx, a) + "\n"
    for rank, h in enumerate(hits, 1):
        c = h.chunk
        score = f"x{int(h.score)}" if lex else f"{h.score:+.3f}"
        txt = h.text[:max_chars].replace("\n", "\n  ")
        body += (f"#{rank} {score}  {h.platform}/{h.partner} msgs[{h.i0}:{h.i1}] "
                 f"{h.n_tokens}tok(est) {c.day}  id={c.id}"
                 + (f"  merged×{len(h.members)}" if len(h.members) > 1 else "") + "\n")
        body += "  " + txt + "\n\n"
    _emit({}, a, body)


def cmd_get(a: argparse.Namespace) -> None:
    idx = load(_index_path(a))
    donor = _donor(idx, a)
    obj = get(idx, a.id, window=a.window, bundle=a.bundle)
    if isinstance(obj, list):
        if not a.pretty:
            return _emit({"donor": donor, "id": a.id, "window": a.window, "msgs": obj}, a)
        return _emit({}, a, "\n".join(m["text"] for m in obj) + "\n")
    if obj is None:
        if not a.pretty:
            return _emit({"donor": donor, "id": a.id, "error": "not found"}, a)
        return _emit({}, a, f"not found: {a.id}\n")
    if not a.pretty:
        return _emit({"donor": donor, **chunk_payload(obj, max_chars=0)}, a)
    _emit({}, a, f"{obj.platform}/{obj.partner} seq{obj.seq} msgs[{obj.i0}:{obj.i1}] "
                 f"{obj.n_tokens}tok {obj.day}\n{obj.text}\n")


# ── push ──────────────────────────────────────────────────────────────────────
def cmd_push(a: argparse.Namespace) -> None:
    idx = load(_index_path(a))
    donor = _donor(idx, a)
    cfg = _cfg(a)
    embedder = _embedder(idx, cfg)
    probes = list(a.probe or [])
    if a.probes_file:
        probes += [ln.strip() for ln in Path(a.probes_file).read_text().splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
    if not probes:
        raise SystemExit("push: need --probe or --probes-file")

    def hits_for(q: str, k: int):
        return search_any(idx, q, embedder, k=k, platform=a.platform, partner=a.partner,
                          since=a.since, until=a.until,
                      max_region_chunks=cfg["retrieve"]["max_region_chunks"])

    p = pack(probes, hits_for, k=a.k or cfg["retrieve"]["k"],
             budget_tokens=a.budget or cfg["retrieve"]["budget"],
             min_threads=a.min_threads or cfg["retrieve"]["min_threads"],
             dedupe=not a.no_dedupe)
    if not a.pretty:
        max_chars = cfg["retrieve"]["max_chars"]
        return _emit({"donor": donor, "kind": idx.kind, "probes": probes, "stats": p.stats,
                      "hits": [hit_payload(h, max_chars=max_chars, text=t)
                               for h, t in zip(p.hits, p.texts)]}, a)
    s = p.stats
    head = (f"{_preamble(idx, a)}  {s['n_hits']} regions · {s['n_msgs']} msgs · {s['tokens']} tok · "
            f"{s['threads']} threads · {s['n_probes']} probes · new/probe={s['per_probe_new']}"
            + (" · BUDGET HIT" if s["budget_hit"] else "")
            + (f" · deduped {s['lines_deduped']} msgs (−{s['tokens_saved_est']} tok est)"
               if s.get("lines_deduped") else ""))
    body = head + "\n\n"
    for h, t in zip(p.hits, p.texts):
        c = h.chunk
        body += (f"{h.platform}/{h.partner} msgs[{h.i0}:{h.i1}] {h.n_tokens}tok(est) {c.day}"
                 + (f" merged×{len(h.members)}" if len(h.members) > 1 else "") + f" id={c.id}\n")
        body += "  " + t[:cfg["retrieve"]["max_chars"]].replace("\n", "\n  ") + "\n\n"
    _emit({}, a, body)


# ── smoke ─────────────────────────────────────────────────────────────────────
def cmd_smoke(a: argparse.Namespace) -> None:
    idx = load(_index_path(a))
    cfg = _cfg(a)
    reranker = _reranker(cfg, enabled=a.rerank)
    results, summary = run_smoke(idx, load_cases(a.cases), k=a.k,
                                embedder=_embedder(idx, cfg), reranker=reranker)
    for r in results:
        mark = f"ok  #{r.rank}" if r.ok else "FAIL"
        print(f"  {mark:8} {r.query!r}" + (f"   {r.detail}" if r.detail else ""))
    print(f"\nsummary  {_preamble(idx, a)}  rerank={summary['rerank']}  "
          f"{summary['passed']}/{summary['n']} passed, {summary['top1']} at rank 1")
    if summary["passed"] != summary["n"]:
        raise SystemExit(1)


# ── loop: model + tools ───────────────────────────────────────────────────────
def cmd_loop(a: argparse.Namespace) -> None:
    idx = load(_index_path(a))
    donor = _donor(idx, a)
    cfg = _cfg(a)
    llm = dict(cfg["llm"])
    if a.model:
        llm["model"] = a.model
    if a.base_url:
        llm["base_url"] = a.base_url
    if a.max_turns:
        llm["max_turns"] = a.max_turns
    cfg = {**cfg, "llm": llm}
    question = a.question or Path(a.question_file).read_text().strip()
    res = run_loop(idx, question, cfg=cfg, embedder=_embedder(idx, cfg),
                   reranker=_reranker(cfg, enabled=a.rerank), push=a.push,
                   pull=not a.no_pull, k=a.k, dry_run=a.dry_run)
    if a.dry_run:
        req = res.calls[0]["dry_run"]
        print(json.dumps({"system": req["messages"][0]["content"],
                          "user": req["messages"][-1]["content"],
                          "tools": [t["function"]["name"] for t in req["tools"]]},
                         ensure_ascii=False, indent=2))
        return
    if a.pretty:
        print(f"donor={donor}  turns={res.turns}  calls={len(res.calls)}  "
              f"injected≈{res.injected_tokens} tok  stopped={res.stopped}")
        for c in res.calls:
            print(f"  {c['tool']}({json.dumps(c['args'], ensure_ascii=False)}) → {c['result_bytes']} B")
        print("\n" + res.answer)
        return
    print(json.dumps({"donor": donor, "question": question, "answer": res.answer,
                      "turns": res.turns, "calls": res.calls,
                      "injected_tokens": res.injected_tokens, "stopped": res.stopped},
                     ensure_ascii=False))


# ── eval (deprecated) ─────────────────────────────────────────────────────────
def cmd_eval(a: argparse.Namespace) -> None:
    raise SystemExit("eval was deleted — use `smoke` (hand-written query → expected pairs)")


def main() -> None:
    p = argparse.ArgumentParser(prog="retrieval", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser, *, index: bool = True) -> None:
        sp.add_argument("--config", help="settings file (default: retrieval/config.yaml; "
                                         "e.g. config.server.yaml)")
        sp.add_argument("--pretty", action="store_true", help="human output instead of JSON")
        if index:
            sp.add_argument("--index", help="index dir (else $RETRIEVAL_INDEX)")
            sp.add_argument("--expect-donor", dest="expect_donor",
                            help="assert the index's donor_id (else $RETRIEVAL_DONOR)")

    def filters(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-k", "--k", type=int, help="hits (default from config)")
        sp.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                        help="override config rerank.enabled")
        sp.add_argument("--platform")
        sp.add_argument("--partner")
        sp.add_argument("--since", help="ISO date YYYY-MM-DD (inclusive)")
        sp.add_argument("--until", help="ISO date YYYY-MM-DD (inclusive)")

    b = sub.add_parser("build", help="chunk + embed a bundle into an index (one index = one donor)")
    common(b, index=False)
    b.add_argument("--bundle", default=DEFAULT_BUNDLE)
    b.add_argument("--out", required=True)
    b.add_argument("--kind", choices=["vector", "lexical"],
                   help="lexical: regex over chunk text, no model (default from config)")
    b.add_argument("--donor-id", dest="donor_id", help="identity stamped into the manifest")
    b.add_argument("--platform", help="only this platform (fb|ig|wa) — debug")
    b.add_argument("--partner", help="only this partner — debug")
    b.add_argument("--limit", type=int, default=0, help="first N threads (debug)")
    b.add_argument("--sample", type=int, default=0, help="N smallest threads (debug)")
    b.set_defaults(func=cmd_build)

    pu = sub.add_parser("push", help="probes → one budgeted context block (the default injection)")
    common(pu)
    filters(pu)
    pu.add_argument("--probe", action="append", help="repeatable probe string")
    pu.add_argument("--probes-file", dest="probes_file", help="one probe per line, # = comment")
    pu.add_argument("--budget", type=int, help="token ceiling (default from config)")
    pu.add_argument("--min-threads", type=int, dest="min_threads",
                    help="distinct partners required in the pack (default from config)")
    pu.add_argument("--no-dedupe", dest="no_dedupe", action="store_true",
                    help="keep repeated overlap messages instead of folding them out")
    pu.set_defaults(func=cmd_push)

    pl = sub.add_parser("pull", help="model-driven retrieval")
    pls = pl.add_subparsers(dest="op", required=True)
    s = pls.add_parser("search", help="ranked chunks for a query")
    common(s)
    filters(s)
    s.add_argument("text")
    s.set_defaults(func=cmd_search)
    g = pls.add_parser("get", help="a chunk by id, optionally ±window raw messages")
    common(g)
    g.add_argument("id", help="chunk id from a search result (the loop maps its ref → id)")
    g.add_argument("--window", type=int, default=0, help="0 = the chunk; >0 = ±window messages")
    g.add_argument("--bundle", help="needed for --window (else from the manifest)")
    g.set_defaults(func=cmd_get)

    m = sub.add_parser("smoke", help="hand-written query → expected pairs (exits 1 on failure)")
    common(m)
    m.add_argument("-k", "--k", type=int, default=5, help="rank cutoff for a pass")
    m.add_argument("--cases", help="JSON list of {query, partner?, day?, contains?}")
    m.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="override config rerank.enabled")
    m.set_defaults(func=cmd_smoke)

    lp = sub.add_parser("loop", help="ask the survey question once: model + tools + state")
    common(lp)
    lp.add_argument("--question", help="the survey question")
    lp.add_argument("--question-file", dest="question_file", help="…or read it from a file")
    lp.add_argument("--push", action="store_true", help="inject a pack before the first call")
    lp.add_argument("--no-pull", dest="no_pull", action="store_true", help="push-only (no tools)")
    lp.add_argument("-k", "--k", type=int, help="hits per search (default from config)")
    lp.add_argument("--model", help="override llm.model")
    lp.add_argument("--base-url", dest="base_url", help="override llm.base_url")
    lp.add_argument("--max-turns", dest="max_turns", type=int, help="override llm.max_turns")
    lp.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None)
    lp.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the request that would be sent, call nothing")
    lp.set_defaults(func=cmd_loop)

    e = sub.add_parser("eval", help="deprecated (synthetic self-retrieval) — use smoke")
    common(e)
    e.set_defaults(func=cmd_eval)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
