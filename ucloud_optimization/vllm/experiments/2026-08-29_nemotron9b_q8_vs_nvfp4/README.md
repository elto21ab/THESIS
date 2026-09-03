# Experiment: Nemotron-Nano-9B-v2 — FP8 (W8A8) vs NVFP4 (W4A4), single B200 MIG slice

**Verdict: NVFP4 wins across the board — +12% throughput, +54% KV capacity. Zero failures both arms.**

## Setup
| | |
|---|---|
| Jobs | 12375750 (FP8) + 12375759 (NVFP4-only), each 1× mig.1g.23gb (20.5 GB, 6 vCPU, 1/7 GPU-h) |
| Engine | vLLM 0.28.0 stable, venv `/work/hpc/venvs/qwen38_vllm` |
| Checkpoints | `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8` (9.6 GB), `-NVFP4` (7.4 GB) — official ModelOpt pair |
| Serve flags | `--port` only (flag-minimal protocol) |
| Workload | `V7_real_12.jsonl` = 12 length-stratified real prompts (~135.7k pp tokens), Q=all-in-flight, temp 0, max_tokens 256 |
| Passes | full (Q=12) + pp-only (max_tokens=1) + tg single-stream (median prompt) |

## Results

| pass | FP8 | NVFP4 | Δ NVFP4 |
|---|---:|---:|---:|
| full ppTPS (Q=12) | 7,425.8 | 8,305.8 | **+11.9%** |
| full tgTPS (Q=12) | 166.5 | 188.0 | **+12.9%** |
| full wall s | 18.28 | 16.34 | −10.6% |
| pp-only ppTPS | 11,200.8 | 12,616.5 | +12.6% |
| tg single-stream tok/s | 67.0 | 80.1 | +19.5% |
| KV pool GiB | 6.16 | 9.46 | **+53.6%** |
| KV tokens | 352,974 | 542,684 | +53.7% |
| max concurrency @131k | 2.69× | 4.14× | +53.9% |
| OK/failed | 12/0 | 12/0 | = |

Notes: tg single-stream includes ~1.9 s prefill (7k tok) → pure decode ≈ 134 (FP8) / 166 (NVFP4) tok/s. Full-pass tgTPS is aggregate over 12 concurrent streams. Hybrid Mamba2 arch → KV pool not the decode bottleneck → throughput deltas are compute-bound (clean quant A/B). MIG-slice numbers: relative deltas trustworthy, absolute values ≈1/7 of full B200 — validate winner on full GPU before publishing absolutes.

## Failure catalogue (each cost real GPU-time — read before relaunch)
1. **`/work` root is EPHEMERAL.** Only attached-folder paths persist (`/work/hpc/*`). All v1 provisioning (venv, 24 GB models, caches under `/work/vLLM`) died with job 12375721. Rule: everything under `/work/hpc/`.
2. **Folder attach requires picking from the dropdown** — free-typing the drive name silently attaches nothing (job 12375744+12375748 burned). Robust path: "Import parameters" from a known-good job.
3. **uv venv quirks**: `source activate` = no-op (PATH unchanged) → use `$VENV/bin/python` + `export PATH=$VENV/bin:$PATH`; no console scripts → shim `bin/vllm` (`python -c "from vllm.entrypoints.cli.main import main…"`) — `python -m vllm` unsupported in 0.28.
4. **kill_serve pattern**: shim cmdline has no literal "vllm serve" → `pkill -f "[v]llm serve"` never matches → server survives → next arm OOMs at startup (2.33 GiB free). Use `pkill -f "entrypoints.cli.main"` + VRAM-release wait.
5. **pkill self-match**: any compound command whose text contains a path/regex matching the kill pattern kills its own shell (exit 255) ×2. Kill via script file piped over scp.
6. **gemma-4-12B FP8 on mig1 does not fit** for full-ctx serving: weights 15 GB → KV pool 5.24 GiB @0.9 < 5.84 GiB required by 262144 max_position_embeddings; util 0.97 = profiling OOM (NVML assert). Switched model to Nemotron 9B. Gemma arm parked for full-B200.
7. Watchers: ssh-only; `pb` unusable inside tmux jobs (env); grep must scope to newest timeline segment (stale-marker false fires ×2).
8. `~/.cache/vllm` (torch.compile AOT) also ephemeral → symlink to `/work/hpc/vLLM/torch_compile_cache`.

## Artifacts
- Remote: `/work/hpc/vLLM/experiments/2026-08-29_nemotron9b_q8_vs_nvfp4/` (jsonl, kv_*.txt, logs, timeline) + `checkpoints_nem9b_0829.tar.gz`
- Local: `ucloud_optimization/vllm/checkpoints/checkpoints_nem9b_0829.tar.gz`
