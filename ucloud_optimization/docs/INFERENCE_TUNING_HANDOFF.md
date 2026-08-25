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

Conclusions: more GPU-resident experts monotonically helped; larger ubatch massively helped PP; exact values do not transfer across architectures. Results checkpoint: `ucloud_optimization/checkpoints/checkpoint-results.tar.gz`.

## New models
1. `LiquidAI/LFM2.5-8B-A1B` (8.3B total / 1.5B active)
2. `LiquidAI/LFM2.5-2.6B` dense

Both should fit fully in 1g.23gb at ~4-bit. Use official LiquidAI Q4_K_M GGUF for llama.cpp. For vLLM/SGLang prefer one shared downloadable native Blackwell-accelerated ~4-bit checkpoint, ideally NVFP4/ModelOpt. Candidate found: `sakamakismile/LFM2.5-8B-A1B-NVFP4`; verify both engines load it. Don't create calibration datasets. Easy calibration-free quant conversion is acceptable; otherwise download native HF quant. Record checkpoint/revision/size/method.

## KV correction
Current llama.cpp has `--kv-unified`: one KV buffer shared across sequences. Use explicitly with `--cont-batching`. vLLM/SGLang still use bounded startup KV arenas, then dynamically page within them. Don't tune vLLM `gpu-memory-utilization` or SGLang `mem-fraction-static` initially; use defaults. Change only to remediate startup/capacity failure.

## Real prompt suites
Original V7 files contain an outer list of independent `[system, assistant, user]` message arrays. Flatten outer list only; preserve each inner array as one request. Three historical model files are duplicates. Tool groups by filename minus target-model suffix and chooses one.

Generated:
- `ucloud_optimization/data/V7_perf_real_48.jsonl`: 48 deterministic length-stratified real requests; short natural categorical outputs.
- `ucloud_optimization/data/V7_perf_reasoning_48.jsonl`: same requests; final user turn asks for >=1,000-word unconstrained analysis; use max 2048 output tokens.
- manifests alongside.

Before paid run, tokenize using actual LFM tokenizer and ideally bucket 12 each at 1–4K, 4–12K, 12–24K, 24–32K tokens. Mixed shuffled lengths deliberately stress shared/dynamic KV allocation.

## Code
- `ucloud_optimization/tools/tune_inference.py`: UV script; resumable JSONL checkpointing, budget watchdog, llama-bench conditional tuner, OpenAI-compatible persistent server concurrency sweep, real chat messages.
- `ucloud_optimization/tools/prepare_benchmark_prompts.py`: duplicate grouping, stratified subset, reasoning transform.
- `ucloud_optimization/configs/inference-tune.example.toml`: template; native engine checkpoint placeholders remain.

Commands:
```bash
uv run ucloud_optimization/tools/tune_inference.py plan --config ucloud_optimization/configs/inference-tune.toml
uv run ucloud_optimization/tools/tune_inference.py run --config ucloud_optimization/configs/inference-tune.toml --budget-min 52
uv run ucloud_optimization/tools/tune_inference.py serve-bench --config ucloud_optimization/configs/inference-tune.toml --engine llama.cpp --concurrency 1 2 4 8 16 32 --prompts ucloud_optimization/data/V7_perf_real_48.jsonl --limit 48
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
