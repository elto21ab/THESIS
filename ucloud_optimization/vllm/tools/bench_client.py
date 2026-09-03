# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Aggregate bench client. All-in-flight or throttled; optional median-prompt selection.

Usage: bench_client.py BASE MODEL SUITE OUT [MAX_TOKENS] [Q] [SELECT]
  MAX_TOKENS default 256; Q default = all rows; SELECT: all|median
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

import httpx


def main() -> int:
    base, model, suite, out_path = sys.argv[1:5]
    max_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    q = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    select = sys.argv[7] if len(sys.argv) > 7 else "all"
    rows = [json.loads(l) for l in open(suite)]
    if select == "median":
        rows = sorted(rows, key=lambda r: r.get("chars", len(r["messages"][0]["content"])))
        rows = [rows[len(rows) // 2]]
    q = q or len(rows)
    results: list[dict] = []
    sem = asyncio.Semaphore(q)
    limits = httpx.Limits(max_connections=q + 4, max_keepalive_connections=q + 4)

    async def one(client: httpx.AsyncClient, r: dict) -> None:
        async with sem:
            t0 = time.perf_counter()
            try:
                resp = await client.post(
                    f"{base}/v1/chat/completions",
                    json={"model": model, "messages": r["messages"],
                          "max_tokens": max_tokens, "temperature": 0},
                    timeout=httpx.Timeout(1800, connect=30),
                )
                dt = time.perf_counter() - t0
                if resp.status_code == 200:
                    u = resp.json().get("usage", {})
                    results.append({"sha": r.get("sha256", ""), "ok": True, "lat_s": dt,
                                    "prompt_tokens": u.get("prompt_tokens", 0),
                                    "completion_tokens": u.get("completion_tokens", 0)})
                else:
                    results.append({"sha": r.get("sha256", ""), "ok": False, "lat_s": dt,
                                    "status": resp.status_code, "err": resp.text[:200]})
            except Exception as e:  # noqa: BLE001
                results.append({"sha": r.get("sha256", ""), "ok": False,
                                "lat_s": time.perf_counter() - t0, "err": f"{type(e).__name__}: {e}"[:200]})

    async def run() -> None:
        async with httpx.AsyncClient(limits=limits) as client:
            await asyncio.gather(*(one(client, r) for r in rows))

    t0 = time.perf_counter()
    asyncio.run(run())
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    pt = sum(r["prompt_tokens"] for r in ok)
    ct = sum(r["completion_tokens"] for r in ok)
    lats = sorted(r["lat_s"] for r in results)
    pct = lambda p: lats[min(len(lats) - 1, int(len(lats) * p))] if lats else 0.0  # noqa: E731
    summary = {
        "model": model, "suite": suite, "select": select, "n": len(rows),
        "Q": q, "ok": len(ok), "failed": len(rows) - len(ok),
        "wall_s": round(wall, 3), "max_tokens": max_tokens,
        "pp_tokens": pt, "tg_tokens": ct,
        "ppTPS": round(pt / wall, 2), "tgTPS": round(ct / wall, 2),
        "lat_p50_s": round(pct(0.5), 3), "lat_p95_s": round(pct(0.95), 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps(summary) + "\n")
    print("SUMMARY", json.dumps(summary))
    return 0 if not summary["failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
