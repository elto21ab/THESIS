# HANDOFF 2026-08-26 — vLLM vs SGLang vs llama.cpp on B200 MIG 1g.23gb
## Session 2 (job: vllm-sglang-offload-test, ~4h, port 2957, ended 02:30)

## Stack (per-job setup — REINSTALL each job)
- B200 MIG 1g.23gb (21GB VRAM), 6 vCPU EPYC 9655, 33.5GB cgroup RAM
- vLLM 0.27.1 (latest, PyPI-confirmed), SGLang 0.5.18, llama.cpp 95b8e33 (SM100 build)
- CUDA 13.0 toolkit: `sudo apt install ninja-build` + keyring + `cuda-toolkit-13-0 libcurand-dev-13-0` (NOT persistent)
- flashinfer JIT cache ~/.cache (NOT persistent, ~9 min rebuild per job)
- Model: LiquidAI/LFM2.5-8B-A1B (8.3B/1.5B active, 24 layers, 32 experts, 4 active/tok, hybrid conv+GQA)
- Persistent: /work/LCPP_OffloadTesting/{venvs,models,configs,data,tools,runs}

## **BREAKTHROUGH: NVFP4 WORKS on SGLang** (last session)
LFM2.5-8B-A1B-NVFP4 (sakamakismile, 6GB) loads + serves on SGLang with:
```bash
export SGLANG_MAMBA_CONV_DTYPE=float16   # fixes causal_conv1d "Dtype [bfloat16] not in allowed [float16]"
venvs/sglang/bin/python -m sglang.launch_server \
  --model-path models/LFM2.5-8B-A1B-NVFP4 --port 8300 --context-length 2048 \
  --mem-fraction-static 0.92 --disable-cuda-graph \
  --mamba-ssm-dtype float16 --disable-flashinfer-autotune \
  --moe-runner-backend flashinfer_cutlass
```
Flags required (each fixes a distinct bug):
1. `SGLANG_MAMBA_CONV_DTYPE=float16` — conv_state buffer hardcodes bf16 (mamba_utils.py:68); env var is the official override. **This is THE fix** — we only had `--mamba-ssm-dtype float16` before (SSM cache ≠ conv state).
2. `--moe-runner-backend flashinfer_cutlass` — NVFP4 MoE rejects default flashinfer_trtllm.
3. `--disable-flashinfer-autotune` — NVML query crashes on MIG (Insufficient Permissions), eager autotune buffer alloc fails.
4. `--disable-cuda-graph` — CUDA graph capture fails on conv_state strides (LFM2 arch).
Last state: server UP (Uvicorn running, "ready to roll"), 6GB weights, 14GB free. Generation request TIMED OUT (90s, no response, server stayed up) — need to debug the first-request hang (possibly slow JIT compile of flashinfer_cutlass MoE kernels on first request; the log froze after Triton '_router_triton_kernel' device-loaded).

### vLLM NVFP4 — still blocked
- `--quantization modelopt_fp4` on sakamakismile checkpoint: AssertionError (merged-column tensor shape, LFM2 conv in_proj). Known vLLM bug #40885 (ModelOpt checkpoint naming, still open).
- On-the-fly `modelopt_fp4`: "Cannot find the config file" (needs pre-quantized checkpoint).
- `nvfp4_per_token`: IndexError dim out of range (same #40885 class).
- `mxfp4`/`mxfp8`: shape mismatch / missing mm_mxfp8 (LFM2 expert layout ≠ vLLM fused-MoE assumption).

### Gemma-4 NVFP4 (unsloth) — blocked by transformers 5.x
- `AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute` — transformers 5.15.1 heterogeneity parser rejects Gemma-4's per-layer head_dim. Not NVFP4, not unsloth. vLLM 0.27.1 requires transformers>=5.5.3 so can't downgrade.
- unsloth README: needs `vllm>=0.25.0 flashinfer-python>=0.6.13 nvidia-cutlass-dsl>=4.5.2` — we're missing nvidia-cutlass-dsl. Install it for Gemma-4.
- Config patch `allow_global_per_layer_attribute_access: true` did NOT help (vLLM reads head_dim via path ignoring it).

### Nemotron-3.5-Lightning-30B-A3B-NVFP4 (nvidia)
- 21GB weights, 52 shards, has hf_quant_config.json. Too big for 1g.23gb MIG (needs full B200/H100).
- Recipe (from HF card, for SGLang): `--mamba-backend flashinfer --mamba-ssm-dtype float16 --enable-mamba-cache-stochastic-rounding --mamba-cache-philox-rounds 5 --mem-fraction-static 0.85 --cuda-graph-max-bs-decode 16 --reasoning-parser nemotron_3`
- Its `--mamba-ssm-dtype float16` + `--mamba-backend flashinfer` pattern is what led us to the LFM2 fix.

## Results recap (LFM2.5-8B-A1B, 16 real prompts, 256 out)
| config | c1 tg | c4 tg | c1 pp |
|---|---|---|---|
| bf16 vLLM | 188.8 | 420.8 | 1409 |
| fp8 vLLM (on-the-fly) | 236.2 | 508.6 | 1687 |
| Q4_K_M llama.cpp | 134.1 | 176.4 | 972 |
| NVFP4 SGLang | ? (server up, req hangs) | ? | ? |
| offload 5% (vLLM prefetch) | 50.1 | 163.2 | 358.7 |
| offload 10% | 29.4 | 107.1 | 210.8 |
| offload 50% | 5.75 | 21.9 | 41.1 |

- fp8 = +25% tg over bf16 (Blackwell fp8 tensor cores). Best quant that works.
- Offload ANY % kills throughput (5% = 3.8x loss). Root cause: vLLM prefetch offloads WHOLE layers (all 32 experts) not router-aware per-expert. llama.cpp -ngl IS router-aware (proven prior session).
- UVA backend (--cpu-offload-gb): hangs flashinfer autotuner on MIG. --load-format mmap: removed in 0.27.1.
- Engine sweep (48 prompts, c1-32): vLLM 2x SGLang at high concurrency (tg 433 vs 208 @ c32), both 48/48. SGLang without graphs at first; with --disable-cuda-graph-padding decode graphs work, gap narrows to 1.5x.
- Prefix caching (144 reqs shared persona): ZERO gain both engines (hybrid arch negates APC/radix).
- Overload (128 reqs/32 slots): 0 failures, graceful queue, latency ∝ queue depth.

## ktransformers (kvcache-ai) — investigation
- NOT a vLLM/SGLang backend plugin. Standalone kt-kernel engine for CPU-GPU heterogeneous inference.
- Has "CPU-GPU Expert Scheduling" (router-aware expert offload) — the smart offload we wanted, but it replaces the engine.
- v0.7.0 (Aug 2026), supports FP8 routed-expert weights, AVX512/AMX CPUs, DeepSeek/GLM/MiniMax/Kimi day0. LFM2/Liquid support UNVERIFIED.
- Relevance: for the too-big model, ktransformers OR llama.cpp (-ngl) are the router-aware offload options; vLLM 0.27.1 prefetch is layer-granular (dumb).

## TODO tomorrow
1. Debug NVFP4 SGLang first-request hang (check if flashinfer_cutlass MoE kernels JIT-compile on first req; try --cuda-graph-max-bs-decode smaller, or warmup request, or watch for OOM in log).
2. If NVFP4 serves: benchmark vs fp8 (expect 2x — NVFP4 is Blackwell native 4-bit).
3. Install nvidia-cutlass-dsl in vllm venv → retry unsloth Gemma-4-12B NVFP4 (needs transformers fix too — maybe try --load-format or a transformers patch; unsloth says "latest transformers" works so maybe upgrade transformers >5.15).
4. Download + try a smaller NVIDIA NVFP4 checkpoint that fits 1g.23gb (Nemotron 30B too big; look for Nemotron-3.5-Lightning-8B or similar).
5. Optional: ktransformers on the big model (CPU-GPU expert scheduling), compare vs llama.cpp -ngl.
6. Check if nvidia-modelopt in vllm venv enables on-the-fly NVFP4 (installed 0.46.0; maybe --quantization modelopt works on bf16 checkpoint with modelopt installed — earlier failed with "config file not found" BEFORE install).

## Artifacts
- runs/lfm25-8b-*/server/*sweep.json (all benchmarks)
- models/: LFM2.5-8B-A1B-bf16 (16G), -NVFP4 (6.5G), -Q4_K_M, gemma-4-12B-NVFP4 (8.7G), gemma-4-26B-A4B-NVFP4 (16G), Nemotron-3.5-Lightning-30B-A3B-NVFP4 (21G)
- venvs/vllm (0.27.1 + nvidia-modelopt 0.46.0), venvs/sglang (0.5.18)
- configs/lfm25-8b-*.toml (all experiment configs)
- tools/tune_inference.py, tools/overload.py
- SGLang patch: venvs/sglang/.../configs/inkling.py (reverted — env var is the real fix)
