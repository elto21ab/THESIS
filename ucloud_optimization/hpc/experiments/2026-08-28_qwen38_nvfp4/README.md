# Experiment: Qwen3.8-27B NVFP4 checkpoint comparison (run 12375523)

**Objective** (per DECISIONS_AND_NEXT_SESSION.md): maximize **aggregate ppTPS/tgTPS per billed GPU-hour**.

## STATUS: STOPPED — zero benchmarks produced. See POSTMORTEM + PRE-FLIGHT CHECKLIST below before any relaunch.

## Job / hardware

| | |
|---|---|
| UCloud job | `12375523` `qwen38-nvfp4-vllm-bench` (2026-08-28, 6h, stopped early ~23:55) |
| Machine | `gpu-nvidia-b200-2-mig.1g` = 2× B200 MIG 1g.23gb (2×20,992 MiB), SM100/CC 10.0 |
| CPU/RAM | 12 vCPU EPYC 9655, 72 GB host RAM |
| Billing | 2/7 GPU-hour per wall-hour (both slices); TP1 single-slice would bill 1/7 |
| Models (downloaded, persisted in `/work/LCPP_OffloadTesting/qwen38/hf/`) | `Inferact/Qwen3.8-27B-NVFP4` (uniform W4A4, 25 GB repo) vs `unsloth/Qwen3.8-27B-NVFP4` (mixed FP8-channel + 4-bit groups, 22 GB repo) |
| Engine | vLLM ≥0.27, venv `/work/LCPP_OffloadTesting/qwen38/venv`; kernel path confirmed `CutlassNvFp4LinearKernel` / `FlashInferCuteDslNvFp4LinearKernel` on sm100 — NVFP4 works on MIG slices |

## POSTMORTEM (2026-08-28, ~1.5 h, 0 benchmarks)

Blockers hit in order — each found at runtime, none pre-checked:

1. **TP1 impossible (planning error).** Inferact ≈24 GiB weights/rank at TP1 > 20.5 GB MIG slice. TP2 mandatory on this tier. Compute `params × bytes/param × (1+overhead)` vs slice VRAM *before* job submission. Note: MIG slices are isolated address spaces — TP2 across slices does per-layer NCCL all-reduce (same physical card, cheap-ish but real). A full 8×B200 node is 8 independent GPUs: for this dense 27B, TP1 × 8 replicas would beat any TP8.
2. **Torch CUDA-allocator NVML assert on MIG** (`NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:1407`): vLLM 0.27 defaults `expandable_segments:true`, which trips NVML-on-MIG. Fix: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` (capital F — this torch build's config parser rejects lowercase; `ValueError: Expected 'True' or 'False'`). Note: this assert also masks plain OOM — read it as "allocator under pressure".
3. **CUDA toolchain was never persisted by the run-12372444 session** (only runtime `lib/`). vLLM JIT-compiles sm100 kernels (vllm ops + flashinfer `sampling`, `fmha_gen`, …) on first serve → needs nvcc + full dev headers. Fixed by apt-installing NVIDIA repo packages; ALL of them now persisted:
   - nvcc/toolkit: `/work/LCPP_OffloadTesting/runtime/cuda-13/` (bin, include×113, lib + `lib64→lib` symlink + `stubs/`)
   - `.deb` cache: `/work/LCPP_OffloadTesting/qwen38/pkg/` (519 MB)
   - packages installed: `cuda-nvcc-13-0 cuda-cudart-{,dev-}13-0 cuda-cccl-13-0 cuda-crt-13-0 cuda-cub-13-0 cuda-thrust-13-0 cuda-toolkit-13-0-config-common libcublas{-dev,}-13-0 cuda-nvrtc-dev-13-0 libcufft-dev-13-0 libcurand-dev-13-0 libcusolver-dev-13-0 libcusparse-dev-13-0 libnvjitlink-dev-13-0`
   - ⚠️ FlashInfer JIT cache lives in `~/.cache/flashinfer` (ephemeral!) — copy to `/work` and point `FLASHINFER_CACHE_DIR`/home there on relaunch to skip recompiles.
   - Install ALL dev packages in ONE apt call — one bad package name aborts the whole transaction silently (cost 20 min).
4. **Operator errors (self-inflicted, documented in UCLOUD_SSH_RUNBOOK.md):**
   - `pkill -f` matching its own compound cmdline (twice) → kills own SSH. Use `pkill -f "[v]llm serve"` in a *separate* ssh call, never in the same command line that contains the literal string.
   - `cp -a ... 2>/dev/null` hid a failed copy for 20 min. Always verify copy results (`ls | wc -l`).
   - `export A=x; ... $A` in one command expands before assignment → empty. Split exports.
   - Remote tmux dies with the SSH session (no linger) → use `setsid nohup ... &`.
   - UCloud "Stop application" raises a native `confirm()` — patchright needs `--accept-dialog`.

**Last known state before stop:** TP2 serve got through weights load + kernel selection; final JIT failure was `cublasLt.h` missing (libcublas-dev apt call had aborted; headers now copied into `runtime/cuda-13/include/`). One relaunch likely suffices to reach HEALTHY.

## PRE-FLIGHT CHECKLIST (next session — execute in order, do not skip)

1. **VRAM math first:** model bytes/rank vs slice VRAM → choose TP. (27B NVFP4 ⇒ TP2 on mig.1g.23gb; TP1 fine on ≥1 full B200.)
2. Restore toolchain from `/work` and VERIFY: `nvcc --version` (with `CUDA_HOME=/work/LCPP_OffloadTesting/runtime/cuda-13`), `ls runtime/cuda-13/include/{cuda_runtime,cublasLt,nvrtc,curand}.h`, `lib64` symlink, `stubs/libcuda.so`.
3. Move flashinfer cache to `/work/LCPP_OffloadTesting/qwen38/flashinfer_cache`; `export FLASHINFER_JIT_CACHE_DIR` (or copy `~/.cache/flashinfer`) so JIT never re-runs.
4. Env for every serve/bench invocation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
   export CUDA_HOME=/work/LCPP_OffloadTesting/runtime/cuda-13
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
   export VLLM_ALLREDUCE_USE_SYMM_MEM=0   # symm-mem fails on MIG, falls back to PYNCCL; suppress noise
   ```
5. Smoke: `vllm serve Inferact/Qwen3.8-27B-NVFP4 --port 8000 --tensor-parallel-size 2` → `curl /health` → kill. Minimal flags only (user preference: no `--max-model-len`; full ctx; let engine manage parallelism; only TP + port).
6. Bench: existing `bench.sh` (rewrite around `tune_inference.py serve-bench` w/ V7 suites `V7_perf_real_48.jsonl` / `V7_perf_reasoning_48.jsonl`, engine config `command = ["vllm","serve","{model}","--port","{port}","--tensor-parallel-size","2","--max-num-seqs","{concurrency}"]`).
7. pkill rule: separate ssh call, bracket pattern.

## Design (unchanged)

Matrix: `{Inferact, unsloth} × {TP2} × {C ∈ 1,2,4,8,16,32} × {V7 real 48, V7 reasoning 48}`. Aggregate prompt/output TPS + wall + failures per C (tuner reports `aggregate_prompt_tps`/`aggregate_output_tps`). MTP pass only after clean baseline (draft acceptance differs per checkpoint: 0.897 vs 0.788 → confound). Optional KV-pool comparison at full ctx (recipe on 2×5090: unsloth 920k vs Inferact 446k tokens).

**Cost note:** TP1 arm abandoned (VRAM, see postmortem #1) → all runs bill 2/7.

## Artifacts

- Remote (persistent): `/work/LCPP_OffloadTesting/qwen38/` — venv, hf/, bench.sh, prov.sh, V7 jsonl, pkg/ (debs), results/, requirements_freeze.txt, logs
- Remote: `/work/LCPP_OffloadTesting/runtime/cuda-13/` — full sm100 toolchain
- Local: `ucloud_optimization/checkpoints/qwen38_checkpoint_0828.tar.gz` (520 MB)
- vLLM version pinned in `requirements_freeze.txt`
