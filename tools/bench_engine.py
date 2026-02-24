# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
#
# Minimal fair LLM inference benchmark engine.
# Engines: llama-completion (native), Ollama (easy to add more CLI engines).
# Fairness: kernel timing, random shuffle, warmup discarded, no caching.
# Usage: importable as a module, or run directly via `uv run bench_engine.py`.

# RESULT: LLAMA.CPP IS THE FASTEST on M1 16GB
# Tested on M1, not yet Xeon Gold 6130: 2sockets×16cores=32physical/64logical (384GB ram)

import os, random, re, subprocess
from abc import ABC, abstractmethod
from shutil import which
from statistics import mean, stdev

_TPS_RE = re.compile(r'(prompt eval|eval) time\s*=\s*[\d.]+\s*ms\s*/\s*\d+.*?\s([\d.]+)\s*tokens per second')


def _prompt(n: int, run_idx: int, prefill: bool) -> str:
    """Unique prompt per run to defeat prefix caching."""
    return f"{run_idx} " + "x " * (n - 1) if prefill else f"W{run_idx}:"


# ── Engine protocol ─────────────────────────────────────────────────

class Engine(ABC):
    name: str

    @abstractmethod
    def run(self, model: str, n_prompt: int, n_gen: int, runs: int, **opts) -> list[float]:
        """Return list of per-run TPS values (kernel timing)."""


class LlamaCli(Engine):
    """Wraps llama-completion for real inference. Kernel timing from stderr, grammar support."""

    def __init__(self, path: str | None = None, grammar: str | None = None, grammar_file: str | None = None):
        self.name = "llama-cpp"
        self.path = path or which("llama-completion")
        self.grammar = grammar            # inline BNF string
        self.grammar_file = grammar_file  # path to .gbnf file

    def run(self, model, n_prompt, n_gen, runs, **opts):
        prefill = n_gen == 0
        mapping = {"threads": "-t", "gpu_layers": "-ngl",
                   "batch_size": "-b", "flash_attn": "-fa"}
        base_flags = ["--no-display-prompt"]
        for k, v in opts.items():
            if k in mapping:
                base_flags.extend([mapping[k], str(int(v))])
        if self.grammar:
            base_flags.extend(["--grammar", self.grammar])
        elif self.grammar_file:
            base_flags.extend(["--grammar-file", self.grammar_file])

        samples = []
        for i in range(runs):
            n = n_prompt if prefill else n_gen
            cmd = [self.path, "-m", model,
                   "-p", _prompt(n, i, prefill),
                   "-n", str(1 if prefill else n)] + base_flags
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            # parse: common_perf_print: prompt eval time = ... XXX.XX tokens per second
            want = "prompt eval" if prefill else "eval"
            tps = 0.0
            for m in _TPS_RE.finditer(proc.stderr):
                if m.group(1) == want:
                    tps = float(m.group(2)); break
            samples.append(tps)
        return samples


class Ollama(Engine):
    """Ollama HTTP API. Uses Ollama's own reported eval durations (kernel-ish)."""

    def __init__(self, url: str = "http://localhost:11434", keep_alive=0, grammar: str | None = None):
        self.name = "ollama"
        self.url = url + "/api/generate"
        self.keep_alive = keep_alive  # 0 = cold reload (fair), "5m" = cached/warm
        self.grammar = grammar        # BNF grammar string, or None for free generation

    def run(self, model, n_prompt, n_gen, runs, **opts):
        import requests
        session = requests.Session()
        prefill = n_gen == 0
        samples = []

        for i in range(runs):
            n = n_prompt if prefill else n_gen
            body = {
                "model": model, "prompt": _prompt(n, i, prefill),
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "num_predict": 1 if prefill else n,
                    "temperature": 0.1,
                    "num_ctx": opts.get("n_ctx", 8192),
                    "num_batch": opts.get("batch_size", 2048),
                    "num_gpu": opts.get("gpu_layers", 99),
                    "num_thread": opts.get("threads", os.cpu_count()),
                },
            }
            if self.grammar:
                body["options"]["grammar"] = self.grammar
            r = session.post(self.url, json=body, timeout=600).json()
            if prefill:
                dur = r.get("prompt_eval_duration", 0)
                tps = r["prompt_eval_count"] / (dur / 1e9) if dur else 0.0
            else:
                dur = r.get("eval_duration", 0)
                tps = r["eval_count"] / (dur / 1e9) if dur else 0.0
            samples.append(tps)

        # unload model after block
        try: session.post(self.url, json={"model": model, "keep_alive": 0})
        except Exception: pass
        session.close()
        return samples


# ── Scheduler ───────────────────────────────────────────────────────

def bench(tasks, sizes, tests=("prefill", "generation"), runs=5, warmup=1, shuffle=True):
    """
    tasks:  [(Engine, model_path_or_name, {hw_opts}), ...]
    shuffle: True for fair cold-start, False for cached/warm mode
    returns: {(engine_name, model, test, n): [tps_samples]}
    """
    schedule = [
        (eng, model, opts, test, n)
        for eng, model, opts in tasks
        for test in tests
        for n in sizes
    ]
    if shuffle:
        random.shuffle(schedule)

    results = {}
    total = len(schedule)
    for i, (eng, model, opts, test, n) in enumerate(schedule, 1):
        label = os.path.basename(model)
        print(f"\r  [{i}/{total}] {eng.name:<12} {label:<40} {test:<10} n={n}   ",
              end="", flush=True)

        p, g = (n, 0) if test == "prefill" else (0, n)
        samples = eng.run(model, p, g, runs=warmup + runs, **opts)
        results[(eng.name, model, test, n)] = samples[warmup:]  # drop warmup

    print()
    return results


# ── Reporter ────────────────────────────────────────────────────────

def report(results, tasks, sizes, tests=("prefill", "generation")):
    """Print mean ± stdev table. Columns = task labels, rows = (test, n)."""
    labels = [f"{e.name}:{os.path.basename(m)}" for e, m, _ in tasks]
    col_w = max(20, max(len(l) for l in labels) + 2)
    hdr = "  ".join(f"{l:>{col_w}}" for l in labels)

    print(f"\n{'test':<12} {'n':>5}  {hdr}")
    print("─" * (19 + len(tasks) * (col_w + 2)))

    for test in tests:
        for n in sizes:
            vals = []
            for eng, model, _ in tasks:
                s = results.get((eng.name, model, test, n), [])
                if len(s) >= 2:
                    vals.append(f"{mean(s):>7.1f} ± {stdev(s):>5.1f}")
                else:
                    vals.append("—")
            print(f"{test:<12} {n:>5}  {'  '.join(f'{v:>{col_w}}' for v in vals)}")


if __name__ == "__main__":
    # TOGGLE HARDWARE: uncomment one
    HW = {"threads": os.cpu_count(), "gpu_layers": 99, "flash_attn": 1, "batch_size": 2048, "n_ctx": 8192}  # Mac M1 16GB
    # HW = {"threads": 32, "gpu_layers": 0, "flash_attn": 0, "batch_size": 2048, "n_ctx": 8192}  # Xeon Gold 6130

    RUNS, WARMUP, SIZES = 3, 1, [128, 256]

    # TOGGLE GRAMMAR: uncomment to constrain generation
    llama = LlamaCli()
    # llama = LlamaCli(grammar_file="json.gbnf")

    # TOGGLE CACHING: keep_alive=0 (cold/fair) vs "5m" (warm). Use shuffle=False with "5m".
    ollama = Ollama()
    # ollama = Ollama(keep_alive="5m")
    # ollama = Ollama(grammar='root ::= ("yes" | "no")')

    # TOGGLE BENCHMARK MODE: uncomment one task list
    # --- A) Engine comparison ---
    tasks = [
        (llama,  "./granite-4.0-350m-base-Q4_K_M.gguf", HW),
        (ollama, "benchgranite",                         HW),
    ]
    # --- B) Quant comparison ---
    # tasks = [
    #     (llama, "./functiongemma-270m-it-Q4_0.gguf",       HW),
    #     (llama, "./functiongemma-270m-it-Q4_K_M.gguf",     HW),
    #     (llama, "./functiongemma-270m-it-IQ4_NL.gguf",     HW),
    #     (llama, "./functiongemma-270m-it-UD-Q4_K_XL.gguf", HW),
    # ]

    # Clean slate: unload any lingering Ollama model
    import requests
    try: requests.post("http://localhost:11434/api/generate", json={"model": "benchgranite", "keep_alive": 0})
    except Exception: pass

    # TOGGLE SHUFFLE: True for cold/fair, False for cached/warm mode
    results = bench(tasks, SIZES, runs=RUNS, warmup=WARMUP)
    # results = bench(tasks, SIZES, runs=RUNS, warmup=WARMUP, shuffle=False)
    report(results, tasks, SIZES)

    print(f"\n  {RUNS} runs + {WARMUP} warmup, kernel TPS")
    print(f"  {HW}")
