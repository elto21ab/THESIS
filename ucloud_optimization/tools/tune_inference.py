# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "psutil>=6"]
# ///
"""Budget-aware, resumable inference tuner.

Optimizes llama.cpp first via conditional search, then validates llama.cpp/vLLM/SGLang
servers under identical request workloads. Every completed trial checkpoints to JSONL.

Examples:
  uv run ucloud_optimization/tools/tune_inference.py plan --config ucloud_optimization/configs/inference-tune.toml
  uv run ucloud_optimization/tools/tune_inference.py run --config ucloud_optimization/configs/inference-tune.toml --budget-min 52
  uv run ucloud_optimization/tools/tune_inference.py report --run-dir ucloud_optimization/runs/lfm25-8b
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses as dc
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil


OOM_PATTERNS = re.compile(
    r"out of memory|failed to create context|failed to allocate|cudaMalloc|OOM|cannot allocate memory",
    re.I,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def geometric(values: list[int]) -> list[int]:
    return sorted(set(values))


@dc.dataclass
class Trial:
    name: str
    phase: str
    engine: str = "llama.cpp"
    flags: dict[str, Any] = dc.field(default_factory=dict)
    status: str = "pending"
    pp_tps: float | None = None
    tg_tps: float | None = None
    wall_s: float | None = None
    error: str | None = None
    started_at: str | None = None
    command: list[str] | None = None

    @property
    def key(self) -> str:
        return json.dumps(
            {"engine": self.engine, "phase": self.phase, "flags": self.flags},
            sort_keys=True,
            separators=(",", ":"),
        )


class State:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "trials.jsonl"
        self.done: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.done[row["key"]] = row

    def save(self, trial: Trial) -> dict[str, Any]:
        row = dc.asdict(trial) | {"key": trial.key, "saved_at": now()}
        append_jsonl(self.path, row)
        self.done[trial.key] = row
        return row


class Budget:
    def __init__(self, minutes: float, reserve_minutes: float):
        self.started = time.monotonic()
        self.total = minutes * 60
        self.reserve = reserve_minutes * 60
        self.durations: list[float] = []

    @property
    def remaining(self) -> float:
        return self.total - (time.monotonic() - self.started)

    def permit(self, estimate: float | None = None) -> bool:
        estimate = estimate or (statistics.median(self.durations[-5:]) if self.durations else 180)
        return self.remaining > self.reserve + estimate


class Monitor:
    """Low-overhead process/cgroup sampler. MIG GPU util may be unavailable."""

    def __init__(self, proc: subprocess.Popen[Any], out: Path):
        self.proc, self.out, self.stop = proc, out, False
        self.samples: list[dict[str, Any]] = []

    async def run(self) -> None:
        p = psutil.Process(self.proc.pid)
        p.cpu_percent(None)
        while not self.stop and self.proc.poll() is None:
            row: dict[str, Any] = {"t": time.time()}
            try:
                tree = [p] + p.children(recursive=True)
                row["cpu_pct"] = sum(x.cpu_percent(None) for x in tree)
                row["rss_bytes"] = sum(x.memory_info().rss for x in tree)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            for file, key in [
                ("/sys/fs/cgroup/memory.current", "cgroup_memory_bytes"),
                ("/sys/fs/cgroup/memory.peak", "cgroup_memory_peak_bytes"),
            ]:
                try:
                    row[key] = int(Path(file).read_text())
                except (OSError, ValueError):
                    pass
            self.samples.append(row)
            await asyncio.sleep(0.5)
        atomic_json(self.out, self.samples)


class Tuner:
    def __init__(self, cfg: dict[str, Any], config_path: Path, budget_min: float | None):
        self.cfg = cfg
        self.model = cfg["model"]
        self.work = cfg["workload"]
        self.search = cfg.get("search", {})
        self.hw = cfg.get("hardware", {})
        self.run_dir = Path(cfg["run"]["dir"])
        self.state = State(self.run_dir)
        minutes = budget_min or float(cfg["run"].get("budget_minutes", 52))
        self.budget = Budget(minutes, float(cfg["run"].get("reserve_minutes", 5)))
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        (self.run_dir / "config.toml").write_bytes(config_path.read_bytes())
        self.llama_bench = Path(cfg["llama"]["bench"])
        self.repetitions = int(self.search.get("screen_repetitions", 1))

    def workload_score(self, pp: float, tg: float) -> float:
        p, g = float(self.work["prompt_tokens"]), float(self.work["output_tokens"])
        seconds = p / pp + g / tg
        fraction = float(self.hw.get("gpu_fraction", 1.0))
        return (p + g) / seconds / fraction

    def completed(self, trial: Trial) -> dict[str, Any] | None:
        row = self.state.done.get(trial.key)
        return row if row and row["status"] == "ok" else None

    async def llama_trial(self, trial: Trial, repetitions: int | None = None) -> dict[str, Any]:
        old = self.completed(trial)
        if old:
            return old
        if not self.budget.permit():
            trial.status, trial.error = "skipped_budget", "insufficient time incl. reserve"
            return self.state.save(trial)

        f = trial.flags
        cmd = [
            str(self.llama_bench), "-m", str(self.model["path"]),
            "-p", str(self.work["prompt_tokens"]), "-n", str(self.work["output_tokens"]),
            "-r", str(repetitions or self.repetitions), "-o", "jsonl",
        ]
        mapping = {
            "threads": "-t", "ngl": "-ngl", "n_cpu_moe": "-ncmoe",
            "batch": "-b", "ubatch": "-ub", "flash_attn": "-fa",
            "cache_k": "-ctk", "cache_v": "-ctv",
        }
        for key, flag in mapping.items():
            if key in f and f[key] is not None:
                cmd += [flag, str(f[key])]
        trial.command, trial.started_at = cmd, now()
        stem = re.sub(r"[^a-zA-Z0-9_.-]", "_", trial.name)
        stdout_path = self.run_dir / "logs" / f"{stem}.jsonl"
        stderr_path = self.run_dir / "logs" / f"{stem}.log"
        started = time.monotonic()
        with stdout_path.open("w") as out, stderr_path.open("w") as err:
            proc = subprocess.Popen(cmd, stdout=out, stderr=err, text=True)
            monitor = Monitor(proc, self.run_dir / "samples" / f"{stem}.json")
            monitor_task = asyncio.create_task(monitor.run())
            try:
                await asyncio.to_thread(proc.wait, timeout=max(1, self.budget.remaining - self.budget.reserve))
            except subprocess.TimeoutExpired:
                proc.send_signal(signal.SIGINT)
                try:
                    await asyncio.to_thread(proc.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                trial.status, trial.error = "timeout", "budget watchdog"
            finally:
                monitor.stop = True
                await monitor_task

        trial.wall_s = time.monotonic() - started
        self.budget.durations.append(trial.wall_s)
        text = stderr_path.read_text(errors="replace")
        if proc.returncode != 0:
            trial.status = "oom" if OOM_PATTERNS.search(text) else "error"
            trial.error = text[-1000:]
            return self.state.save(trial)

        rows = [json.loads(x) for x in stdout_path.read_text().splitlines() if x.strip()]
        pp = [float(x["avg_ts"]) for x in rows if int(x.get("n_prompt", 0)) > 0]
        tg = [float(x["avg_ts"]) for x in rows if int(x.get("n_gen", 0)) > 0]
        if not pp or not tg:
            trial.status, trial.error = "error", "missing pp/tg result"
        else:
            trial.status, trial.pp_tps, trial.tg_tps = "ok", statistics.mean(pp), statistics.mean(tg)
        return self.state.save(trial)

    async def placement(self, base: dict[str, Any]) -> list[dict[str, Any]]:
        """MoE: binary-search minimum fitting CPU expert-layer count."""
        total = int(self.model.get("moe_layers", 0))
        if not total:
            return []
        rows: list[dict[str, Any]] = []
        lo, hi = 0, total
        # Endpoints + binary boundary. `lo` known failing only after first failure.
        candidates = [total, 0]
        tested: set[int] = set()
        while candidates and self.budget.permit():
            n = candidates.pop(0)
            if n in tested:
                continue
            tested.add(n)
            row = await self.llama_trial(Trial(
                f"place_ncmoe_{n}", "placement", flags=base | {"ngl": 999, "n_cpu_moe": n}
            ))
            rows.append(row)
            if n == 0:
                if row["status"] == "ok":
                    return rows
                lo, hi = 0, total
                candidates.append((lo + hi) // 2)
                continue
            if row["status"] == "ok":
                hi = min(hi, n)
            else:
                lo = max(lo, n)
            if hi - lo > 1:
                candidates.append((lo + hi) // 2)
        return rows

    async def run(self) -> None:
        threads = int(self.hw.get("cpus", os.cpu_count() or 1))
        base = {
            "threads": threads, "batch": 2048, "ubatch": 512,
            "flash_attn": "on",
        }
        await self.llama_trial(Trial("default", "baseline", flags={}))

        placement_rows = await self.placement(base) if self.model.get("class") == "moe" else []
        valid = [x for x in placement_rows if x["status"] == "ok"]
        if valid:
            # Fastest measured placement; boundary typically wins.
            best = max(valid, key=lambda x: self.workload_score(x["pp_tps"], x["tg_tps"]))
            base |= {"ngl": 999, "n_cpu_moe": best["flags"]["n_cpu_moe"]}
        else:
            base |= {"ngl": 999}

        # Geometric ubatch; early stop after OOM or < threshold gain.
        previous: dict[str, Any] | None = None
        for ub in geometric(self.search.get("ubatches", [512, 1024, 2048, 4096])):
            row = await self.llama_trial(Trial(f"ubatch_{ub}", "ubatch", flags=base | {
                "ubatch": ub, "batch": max(2048, ub)
            }))
            if row["status"] != "ok":
                break
            if previous:
                gain = row["pp_tps"] / previous["pp_tps"] - 1
                if gain < float(self.search.get("min_gain", 0.05)):
                    break
            previous = row
        if previous:
            base |= {"ubatch": previous["flags"]["ubatch"], "batch": previous["flags"]["batch"]}

        # One lower thread point; retain winner.
        thread_rows = []
        for t in sorted(set([max(1, threads // 2), threads])):
            thread_rows.append(await self.llama_trial(Trial(
                f"threads_{t}", "threads", flags=base | {"threads": t}
            )))
        valid = [x for x in thread_rows if x["status"] == "ok"]
        if valid:
            best = max(valid, key=lambda x: self.workload_score(x["pp_tps"], x["tg_tps"]))
            base["threads"] = best["flags"]["threads"]

        # Robust final measurement; server phase is separate because it needs real prompts.
        await self.llama_trial(Trial("winner", "validation", flags=base), repetitions=int(self.search.get("final_repetitions", 3)))
        self.write_report()

    def write_report(self) -> None:
        rows = [x for x in self.state.done.values() if x["status"] == "ok" and x.get("pp_tps")]
        rows.sort(key=lambda x: self.workload_score(x["pp_tps"], x["tg_tps"]), reverse=True)
        lines = [
            "# Inference tuning report", "",
            f"Model: `{self.model['path']}`", "",
            "| Rank | Trial | Phase | PP tok/s | TG tok/s | Workload-weighted tok/s / GPU fraction | Wall s |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
        for i, x in enumerate(rows, 1):
            score = self.workload_score(x["pp_tps"], x["tg_tps"])
            lines.append(f"| {i} | {x['name']} | {x['phase']} | {x['pp_tps']:.2f} | {x['tg_tps']:.2f} | {score:.2f} | {x['wall_s']:.1f} |")
        (self.run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
        if rows:
            atomic_json(self.run_dir / "best.json", rows[0])


async def wait_ready(url: str, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"server exited {process.returncode}")
            try:
                if (await client.get(url + "/v1/models")).status_code < 500:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError("server startup timeout")


async def one_request(client: httpx.AsyncClient, url: str, model: str, item: dict[str, Any], output_tokens: int) -> dict[str, Any]:
    t0 = time.monotonic()
    if "messages" in item:
        endpoint = "/v1/chat/completions"
        payload = {"model": model, "messages": item["messages"], "max_tokens": output_tokens,
                   "temperature": 0, "stream": False}
    else:
        endpoint = "/v1/completions"
        payload = {"model": model, "prompt": item["prompt"], "max_tokens": output_tokens,
                   "temperature": 0, "stream": False}
    response = await client.post(url + endpoint, json=payload)
    response.raise_for_status()
    body = response.json()
    return {"elapsed_s": time.monotonic() - t0, **body.get("usage", {})}


async def _measure_server(base: str, model: str, concurrency: int, prompts: list[dict[str, Any]], output_tokens: int, request_timeout: float) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(timeout=request_timeout, limits=limits) as client:
        sem = asyncio.Semaphore(concurrency)
        async def run(item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await one_request(client, base, model, item, output_tokens)
        t0 = time.monotonic()
        results = await asyncio.gather(*(run(p) for p in prompts), return_exceptions=True)
        wall = time.monotonic() - t0
    ok = [x for x in results if isinstance(x, dict)]
    prompt_toks = sum(int(x.get("prompt_tokens", 0)) for x in ok)
    output_toks = sum(int(x.get("completion_tokens", 0)) for x in ok)
    latencies = sorted(float(x["elapsed_s"]) for x in ok)
    pct = lambda p: latencies[min(len(latencies)-1, round((len(latencies)-1)*p))] if latencies else None
    return {
        "concurrency": concurrency, "wall_s": wall, "requests_ok": len(ok),
        "requests_failed": len(results)-len(ok), "prompt_tokens": prompt_toks,
        "output_tokens": output_toks, "aggregate_prompt_tps": prompt_toks/wall,
        "aggregate_output_tps": output_toks/wall, "request_latency_p50_s": pct(.5),
        "request_latency_p95_s": pct(.95),
    }


async def bench_server_sweep(cfg: dict[str, Any], engine_name: str, concurrencies: list[int], prompts: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    """Launch once at max capacity; sweep client concurrency without reloading weights."""
    ecfg = cfg["engines"][engine_name]
    port = int(ecfg.get("port", 8100)); model = str(ecfg["model"])
    max_c = max(concurrencies)
    replacements = {"{concurrency}": str(max_c), "{port}": str(port), "{model}": model}
    cmd = [replacements.get(str(x), str(x)) for x in ecfg["command"]]
    base = f"http://127.0.0.1:{port}"; run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir/f"{engine_name}-persistent.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    rows: list[dict[str, Any]] = []
    try:
        await wait_ready(base, proc, float(ecfg.get("startup_timeout", 900)))
        # Unmeasured warmup. Use shortest item; max 16 output tokens.
        await _measure_server(base, model, 1, prompts[:1], min(16, int(cfg["workload"]["output_tokens"])), float(ecfg.get("request_timeout", 3600)))
        previous_rate: float | None = None
        for c in concurrencies:
            row = await _measure_server(base, model, c, prompts, int(cfg["workload"]["output_tokens"]), float(ecfg.get("request_timeout", 3600)))
            row |= {"engine": engine_name, "command": cmd, "saved_at": now()}
            atomic_json(run_dir/f"{engine_name}-c{c}.json", row); rows.append(row)
            # End only on failures. Throughput can temporarily flatten then rise again.
            if row["requests_failed"]:
                break
            previous_rate = (row["prompt_tokens"]+row["output_tokens"])/row["wall_s"]
        atomic_json(run_dir/f"{engine_name}-sweep.json", rows)
        return rows
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGINT)
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
        log.close()


async def bench_server(cfg: dict[str, Any], engine_name: str, concurrency: int, prompts: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Launch any OpenAI-compatible engine and measure real aggregate throughput.

    Config command examples:
      llama.cpp: ["llama-server", "-m", "...", "--kv-unified", ...]
      vllm:       ["vllm", "serve", "...", "--max-num-seqs", "{concurrency}"]
      sglang:     ["python", "-m", "sglang.launch_server", "--model-path", "..."]
    """
    ecfg = cfg["engines"][engine_name]
    port = int(ecfg.get("port", 8100))
    model = str(ecfg["model"])
    replacements = {"{concurrency}": str(concurrency), "{port}": str(port), "{model}": model}
    cmd = [replacements.get(str(x), str(x)) for x in ecfg["command"]]
    base = f"http://127.0.0.1:{port}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = (run_dir / f"{engine_name}-c{concurrency}.log").open("w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        await wait_ready(base, proc, float(ecfg.get("startup_timeout", 900)))
        limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
        async with httpx.AsyncClient(timeout=float(ecfg.get("request_timeout", 3600)), limits=limits) as client:
            sem = asyncio.Semaphore(concurrency)
            async def run(item: dict[str, Any]) -> dict[str, Any]:
                async with sem:
                    return await one_request(client, base, model, item, int(cfg["workload"]["output_tokens"]))
            t0 = time.monotonic()
            results = await asyncio.gather(*(run(p) for p in prompts), return_exceptions=True)
            wall = time.monotonic() - t0
        ok = [x for x in results if isinstance(x, dict)]
        prompt_toks = sum(int(x.get("prompt_tokens", 0)) for x in ok)
        output_toks = sum(int(x.get("completion_tokens", 0)) for x in ok)
        row = {
            "engine": engine_name, "concurrency": concurrency, "wall_s": wall,
            "requests_ok": len(ok), "requests_failed": len(results) - len(ok),
            "prompt_tokens": prompt_toks, "output_tokens": output_toks,
            "aggregate_prompt_tps": prompt_toks / wall,
            "aggregate_output_tps": output_toks / wall,
            "command": cmd, "saved_at": now(),
        }
        atomic_json(run_dir / f"{engine_name}-c{concurrency}.json", row)
        return row
    finally:
        os.killpg(proc.pid, signal.SIGINT)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        log.close()


def load_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            x = json.loads(line)
            if "messages" in x:
                prompts.append({"messages": x["messages"]})
            elif x.get("prompt") or x.get("text"):
                prompts.append({"prompt": x.get("prompt") or x["text"]})
        else:
            prompts.append({"prompt": line})
        if len(prompts) >= limit:
            break
    if not prompts:
        raise ValueError("prompt file needs text lines or JSONL messages/prompt/text fields")
    return prompts


def plan(cfg: dict[str, Any]) -> None:
    model = cfg["model"]
    print(f"model={model['path']} class={model['class']}")
    print("1 default baseline")
    if model["class"] == "moe":
        layers = int(model.get("moe_layers", 0))
        print(f"~{math.ceil(math.log2(layers)) + 2} placement boundary trials max" if layers else "placement search skipped until moe_layers is set")
    print(f"ubatch geometric: {cfg.get('search', {}).get('ubatches', [512,1024,2048,4096])}")
    print("threads: half, all")
    print("winner: final repetitions")
    print("server comparison: each engine × configured concurrency; separate command")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    for action in ("plan", "run"):
        p = sub.add_parser(action); p.add_argument("--config", type=Path, required=True)
        if action == "run": p.add_argument("--budget-min", type=float)
    p = sub.add_parser("report"); p.add_argument("--run-dir", type=Path, required=True)
    p = sub.add_parser("serve-bench")
    p.add_argument("--config", type=Path, required=True); p.add_argument("--engine", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1,2,4,8]); p.add_argument("--prompts", type=Path, required=True); p.add_argument("--limit", type=int, default=32)
    args = ap.parse_args()
    if args.action == "report":
        cfg = tomllib.loads((args.run_dir / "config.toml").read_text())
        Tuner(cfg, args.run_dir / "config.toml", 1).write_report(); return 0
    cfg = tomllib.loads(args.config.read_text())
    if args.action == "plan": plan(cfg); return 0
    if args.action == "run": asyncio.run(Tuner(cfg, args.config, args.budget_min).run()); return 0
    prompts = load_prompts(args.prompts, args.limit)
    out = Path(cfg["run"]["dir"]) / "server"
    asyncio.run(bench_server_sweep(cfg, args.engine, args.concurrency, prompts, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
