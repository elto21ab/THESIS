# UCloud inference tuning — handoff

## Objective
Minimize billed GPU-hours for mixed long-prompt inference while reporting prefill TPS, generation TPS, end-to-end wall time, concurrency scaling, CPU/RAM, VRAM allocation, and failures. Primary engine llama.cpp; vLLM/SGLang only need enough tests to detect a material (>15%) serving advantage.

## UCloud hardware verified
Smallest B200 tier:
- `gpu-nvidia-b200-1-mig.1g`
- B200 MIG `1g.23gb`; CUDA reports 20,992 MiB usable; CC 10.0 / SM100
- 6 vCPU quota, AMD EPYC 9655; 36 GB decimal RAM (cgroup ~33.5 GiB)
- billing 1/7 GPU-hour per wall hour (not 1/8 compute; memory is ~1/8 full B200)
- MIG per-slice SM/memory util unavailable through `nvidia-smi`; don't invent it. CPU UI correctly ranges 0–600%.

## Previous Qwen3.6 experiment
llama.cpp commit `c4b0225`, CUDA 13.0, Release, `CMAKE_CUDA_ARCHITECTURES=100`, `-j6`. Qwen3.6-35B-A3B Bartowski Q4_K_M, 20,480 prompt + 512 generation tokens, FA explicitly on.

| CPU MoE layers | ubatch | PP tok/s | TG tok/s |
|---:|---:|---:|---:|
| 40 | 512 | 488.60 | 46.69 |
| 30 | 512 | 556.51 | 53.22 |
| 20 | 512 | 655.55 | 60.17 |
| 10 | 512 | 787.79 | 62.18 |
| 7 | 512 | 850.36 | 66.64 |
| 5 | 512 | 896.88 | 70.34 |
| 0 | 512 | context init failed/OOM | — |
| 10 | 256 | 539.66 | 62.27 |
| 10 | 1024 | 1081.37 | 61.51 |

Conclusions: more GPU-resident experts monotonically helped; larger ubatch massively helped PP; exact values do not transfer across architectures. Results checkpoint: `ucloud_checkpoint/checkpoint-results.tar.gz`.

## New models
1. `LiquidAI/LFM2.5-8B-A1B` (8.3B total / 1.5B active)
2. `LiquidAI/LFM2.5-2.6B` dense

Both should fit fully in 1g.23gb at ~4-bit. Use official LiquidAI Q4_K_M GGUF for llama.cpp. For vLLM/SGLang prefer one shared downloadable native Blackwell-accelerated ~4-bit checkpoint, ideally NVFP4/ModelOpt. Candidate found: `sakamakismile/LFM2.5-8B-A1B-NVFP4`; verify both engines load it. Don't create calibration datasets. Easy calibration-free quant conversion is acceptable; otherwise download native HF quant. Record checkpoint/revision/size/method.

## KV correction
Current llama.cpp has `--kv-unified`: one KV buffer shared across sequences. Use explicitly with `--cont-batching`. vLLM/SGLang still use bounded startup KV arenas, then dynamically page within them. Don't tune vLLM `gpu-memory-utilization` or SGLang `mem-fraction-static` initially; use defaults. Change only to remediate startup/capacity failure.

## Real prompt suites
Original V7 files contain an outer list of independent `[system, assistant, user]` message arrays. Flatten outer list only; preserve each inner array as one request. Three historical model files are duplicates. Tool groups by filename minus target-model suffix and chooses one.

Generated:
- `data/04_batch/prompts/V7_perf_real_48.jsonl`: 48 deterministic length-stratified real requests; short natural categorical outputs.
- `data/04_batch/prompts/V7_perf_reasoning_48.jsonl`: same requests; final user turn asks for >=1,000-word unconstrained analysis; use max 2048 output tokens.
- manifests alongside.

Before paid run, tokenize using actual LFM tokenizer and ideally bucket 12 each at 1–4K, 4–12K, 12–24K, 24–32K tokens. Mixed shuffled lengths deliberately stress shared/dynamic KV allocation.

## Code
- `tools/tune_inference.py`: UV script; resumable JSONL checkpointing, budget watchdog, llama-bench conditional tuner, OpenAI-compatible persistent server concurrency sweep, real chat messages.
- `tools/prepare_benchmark_prompts.py`: duplicate grouping, stratified subset, reasoning transform.
- `configs/inference-tune.example.toml`: template; native engine checkpoint placeholders remain.

Commands:
```bash
uv run tools/tune_inference.py plan --config configs/inference-tune.toml
uv run tools/tune_inference.py run --config configs/inference-tune.toml --budget-min 52
uv run tools/tune_inference.py serve-bench --config configs/inference-tune.toml --engine llama.cpp --concurrency 1 2 4 8 16 32 --prompts data/04_batch/prompts/V7_perf_real_48.jsonl --limit 48
```
`serve-bench` now launches one persistent server at max configured concurrency, warms once, sweeps client concurrency, then exits. It checkpoints each concurrency result. It currently reports usage-based aggregate prompt/output TPS and request latency; streaming TTFT/ITL remains TODO.

## Next-run methodology
Provision Terminal Ubuntu, 1 hour, 1/7 B200, SSH. Correctly mount/create persistent `LCPP_OffloadTesting` before submit. Avoid prior mistake where typed folder name wasn't actually selected and no resource mounted.

Preflight before paid benchmark clock if possible:
- upload scripts/prompts/config
- install CUDA/build llama.cpp SM100 or restore saved build/runtime
- install vLLM/SGLang envs
- download all model artifacts
- smoke-load every engine/checkpoint

Hour budget:
1. llama.cpp 8B: default vs explicit `ngl all`, FA on, unified KV; ubatch 512→1024→2048 (stop OOM/<5% gain); persistent concurrency 1→32 on real + reasoning suites.
2. llama.cpp 2.6B: default/full GPU; ubatch 1024→2048→4096; concurrency up to 64; real + reasoning.
3. vLLM 8B: native quant default + concurrency sweep. One batch-token adjustment only if clearly underused.
4. SGLang 8B: same checkpoint/default + concurrency sweep. One adjustment only if clearly underused.
5. 2.6B alternatives only if time.
6. Reserve final 5 min for validation/report/SCP.

Primary metrics: PP TPS, TG TPS, wall time, req/s, output-token counts, failures, max useful concurrency, tokens per billed GPU-hour. TPS alone is insufficient when reasoning output lengths differ. Migration gate: <15% end-to-end throughput/GPU-hour improvement means keep llama.cpp unless KV stability/capacity is materially better.

## Scaling
Don't directly assume linear 1/7→8 B200. Current→8 full GPUs is ~56x compute, ~63x VRAM, 64x CPU/RAM. Treat 56x as optimistic ceiling. For models fitting one GPU, prefer independent replicas; validate 1 full B200 then 2 replicas before node-scale forecast.

## TODO before next UCloud submit
1. Add tokenizer-based exact token buckets.
2. Add SSE streaming parser for TTFT + inter-token latency.
3. Add autonomous server-config stage scheduler (llama ubatch/default variants, then engine gates) under one global budget.
4. Determine exact official GGUF filenames/revisions and native NVFP4 support in current vLLM/SGLang.
5. Write bootstrap script with persistent mount verification (`JobParameters.json` resources nonempty + folder contents survive).
6. Test CLI locally with a tiny mock/OpenAI server or tiny model.

## Active run 12372444 exact reproducibility (2026-08-23)
Persistent mount: `/work/LCPP_OffloadTesting` on UCloud filesystem (`df` filesystem `ucloud`). Job expiry 16:50 local.

Exact stack:
- Ubuntu 24.04 container; Linux `6.12.0-211.40.1.el10_2.x86_64`
- llama.cpp commit `95b8e33e16bb9a60de780a70930ebf729db6a90a`, server reports `0.2.0-dev`, build 1
- CUDA compiler 13.0, nvcc `V13.0.88`; `cuda-compiler-13-0=13.0.3-1`, cudart `13.0.96-1`
- cuBLAS `13.1.1.3-1`
- GCC `13.3.0`; CMake `3.28.3`; Ninja `1.11.1`
- build: Release CUDA, `CMAKE_CUDA_ARCHITECTURES=100`, six jobs
- hardware: B200 MIG 1g.23gb; CPU cgroup `600000/100000` = 6 cores; RAM cgroup `35,999,997,952` bytes

Build command:
```bash
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
cmake -S src/llama.cpp -B src/llama.cpp/build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX"
cmake --build src/llama.cpp/build -j6
```

Persistent artifacts:
- binary/source/build: `src/llama.cpp`, `src/llama.cpp/build`
- portable launcher: `bin/llama-server-sm100`
- copied CUDA runtime: `runtime/cuda-13/lib` (cudart, cuBLAS, cuBLASLt; 569 MB)
- manifests: `manifests/environment.txt`, `cuda-packages.tsv`, `llama-server.ldd.txt`, `models.sha256`
- model: official `LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q4_K_M.gguf`; SHA-256 `4923ec14f06b968b74d663e5949867d2d9c3bf13a20b8be1a9f9af39989b2bb0`
- tests/logs: `runs/`; inputs: `data/`; code/config: `tools/`, `configs/`

Fresh container execution still requires compatible NVIDIA driver (`libcuda.so.1` comes from host). The wrapper supplies persisted CUDA user-space runtime. Reinstall compiler only when rebuilding. Verify:
```bash
cd /work/LCPP_OffloadTesting
sha256sum -c manifests/models.sha256
bin/llama-server-sm100 --version
ldd src/llama.cpp/build/bin/llama-server | grep 'not found' && exit 1 || true
```

## Active-run final additions
- vLLM `0.27.1`; SGLang `0.5.18`; persistent isolated envs `venvs/vllm`, `venvs/sglang`.
- Downloaded complete `sakamakismile/LFM2.5-8B-A1B-NVFP4` repo; ModelOpt producer `0.44.0`. See `runs/ALT_ENGINES.md` and smoke logs.
- 2.6B official Q4_K_M exact size `1,674,455,040`; compact short/medium test ub2048/256 outputs: C1 22.005s, C2 16.449s, C4 16.452s; all 8/8. C2 best throughput; C4 no gain + p95 10.12s vs 5.43s.
- Concurrency result JSON stores requested C, successes/failures, token totals, wall, p50/p95. TODO: serialize failed HTTP status/body and parse queue/preemption engine metrics.
