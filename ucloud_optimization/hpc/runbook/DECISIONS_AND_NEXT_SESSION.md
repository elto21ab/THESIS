# UCloud inference optimization — decision checkpoint

Read this first in a new session, then `INFERENCE_TUNING_HANDOFF.md`.

## Objective
Maximize completed batch workload per billed GPU-hour on UCloud B200 hardware. Primary engine: llama.cpp. vLLM/SGLang only matter if they beat llama.cpp materially (>15% end-to-end throughput/cost or materially better reliable capacity). Models: `LiquidAI/LFM2.5-8B-A1B` and `LiquidAI/LFM2.5-2.6B`; previous Qwen3.6-35B-A3B results remain useful as historical MoE/offload evidence.

## User priorities
Primary report columns, in this order:

| Engine/Model/Quant/GB | P | Q | ctx pool | KV type | ubatch | ppTPS | tgTPS | wall | OK | failed/retried | failure behavior | full flags |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|

TPS is total across all successful concurrent streams, never per-stream. Any TPS from a partial-success pass must be marked invalid for winner selection. Optimize full-dataset wall time (incl. retries).

## Terminology
- `P`: server active-slot ceiling = llama.cpp `--parallel`.
- `Q`: client outstanding HTTP requests/backlog window. Operational only; don't tune as a performance dimension. Use roughly `Q=min(4P,32)` so server stays fed while limiting ambiguous in-flight work after failure.
- `ctx-size`: total unified/shared KV cell pool when `--kv-unified` is enabled; not per-request context.
- Model max context: maximum one request may use. Distinct from aggregate shared pool.
- `p95 s`: 95th percentile request latency in seconds. Secondary for offline batch; zero loss + total wall dominate.
- SLA: service-level constraint, e.g. failure rate 0 or p95 under a threshold.

## Explicit rejection: token-aware admission
Do **not** gate admission using estimated prompt length or reserve `prompt_tokens + max_tokens` per request. User wants every admitted request free to use maximum model context if the model/runtime deems it fit. Safety/control strategy instead:

1. Size aggregate `ctx-size` and `P` experimentally.
2. Keep finite Q for checkpoint/backpressure only.
3. Preserve every request by stable ID.
4. On retryable capacity failure, requeue unchanged request and reduce effective client concurrency/active pressure.
5. Never truncate a prompt or lower its max context to make it fit unless explicitly requested.

This means worst-case guarantee for P simultaneous max-context requests mathematically requires approximately `P × model_max_context` shared cells. A smaller pool is allowed only as an empirical multiplexing choice; failures must be retried, never dropped.

## Q vs P
If `Q=30`, `P=10`: normally 10 active server slots + up to 20 requests waiting in client/server/HTTP queues. `Q>P` should not itself fail; failure can occur from HTTP timeout/queue limits, process crash, or KV exhaustion after admission. For offline batches, all records remain in a durable pending queue; only Q are outstanding. P is the compute/memory optimization knob.

Important correction about run 12372444: server P stayed fixed at the maximum concurrency ladder value while client C/Q varied. Reasoning used P4 with client concurrency 1→2→4. Mixed-real used P8 with client concurrency 1→2→4→8. C2 had 16/16 success and no confirmed failure/queue problem; it merely gave no throughput gain. Earlier wording `queued/interleaved` as failure behavior was wrong.

## Failure/retry state machine
Implement durable SQLite or append-only JSONL:

```text
pending → running → succeeded
                  ↘ failed_retryable → pending
                  ↘ failed_terminal
```

Store request ID/SHA-256, source, attempt, P, Q, ctx-size, KV type, ubatch, HTTP status, response body, exception class, elapsed time, input/output usage. Retry policy for retryable KV/HTTP 429/5xx/timeout/reset:

```text
initial pass at chosen pressure
failed subset → retry with effective concurrency halved
repeat down to 1
still fails → terminal report
```

Server `--parallel` may remain at max while client lowers effective pressure. Do not silently discard failed prompts.

## What happened in run 12372444
Hardware: B200 MIG 1g.23gb, 20,992 MiB usable VRAM, 6 EPYC 9655 CPU quota, ~36 GB RAM, billed 1/7 GPU-hour/hour.

Exact stack:
- llama.cpp commit `95b8e33e16bb9a60de780a70930ebf729db6a90a`
- CUDA 13.0, nvcc V13.0.88; cuDART 13.0.96; cuBLAS 13.1.1.3
- SM100 Release build; GCC 13.3; CMake 3.28.3; Ninja 1.11.1
- vLLM 0.27.1; SGLang 0.5.18

### 8B clean ubatch result
16 mixed-length requests, 184,200 input tokens + 8,192 output tokens; Q4_K_M, full GPU, FA, unified KV, continuous batching, P1, 128K shared pool, default KV type, no prompt cache.

| ubatch | ppTPS | tgTPS | wall | p95 | OK/fail |
|---:|---:|---:|---:|---:|---:|
|512|1847.30|82.16|99.713s|12.60s|16/0|
|1024|2029.85|90.27|90.746s|10.95s|16/0|
|2048|2118.06|94.20|86.966s|10.25s|16/0|
|2048 validation|2117.59|94.18|86.985s|10.26s|16/0|

2048 validates at 0.02% wall drift. 2048 vs 512 = 12.8% lower wall. 2048 vs 1024 = 4.16% gain, below 5% geometric-search threshold. 2048 max-speed winner; 1024 safer concurrency runner-up.

### 8B capacity failure
At higher concurrency under 128K shared pool, logs confirm runtime KV exhaustion:

```text
failed to find available cells in kv cache
failed to prepare attention ubatches
failed to find a memory slot for batch of size ...
failed to find free space in the KV cache
failed to restore state / prompt
```

llama.cpp tried smaller internal batch fragments (2048→1024→...→256) but did not change configured ubatch or enlarge KV. Server survived; some requests failed. Reasoning C4: 12/16. Mixed-real C8: 8/16. C2 reasoning: 16/16, no throughput gain; cause not isolated, don't claim KV waiting.

### 2.6B partial result
Short/medium subset only, 8 requests, ubatch 2048, 256 outputs:

| client C | ppTPS | tgTPS | wall | p95 | OK/fail |
|---:|---:|---:|---:|---:|---:|
|1|939.56|93.07|22.005s|3.22s|8/0|
|2|1256.89|124.50|16.449s|5.43s|8/0|
|4|1256.68|124.48|16.452s|10.12s|8/0|

C2 won this partial workload; C4 added no throughput and doubled p95. Must rerun equal mixed workload before cross-model claims.

### Alternative engines
Downloaded complete `sakamakismile/LFM2.5-8B-A1B-NVFP4` HF repo: weights + config + `hf_quant_config.json` + tokenizer + chat template + generation config + run docs. ModelOpt producer 0.44.0, NVFP4 W4A4. Unsloth had no target LFM2.5 NVFP4 repos; only native/BF16 and GGUF.

- vLLM 0.27.1 recognized LFM2 MoE/ModelOpt FP4 but failed merged-column tensor shape assertion while loading. Artifact docs target vLLM 0.21.0 → pin 0.21.0 once.
- SGLang 0.5.18 reached scheduler then causal-conv JIT rejected BF16; kernel allowed FP16 only → smoke once with `--dtype float16`.
- Neither served; no throughput/KV comparison exists.

## Manual llama.cpp flags in run

```bash
llama-server \
  -m MODEL.gguf \
  --port 8100 \
  --parallel P \
  --kv-unified \
  --cont-batching \
  --ctx-size 128000 \
  --ubatch-size 512|1024|2048 \
  --threads 6 \
  --threads-batch 6 \
  --flash-attn on \
  --no-cache-prompt \
  --metrics \
  --no-webui
```

API: temperature 0, non-streaming, max tokens 32/256/512. Logical batch default 2048. GPU placement/default fit was implicit; model was fully GPU-resident. KV type was not manually specified.

## Next-run policy
Use Q8 KV as the main capacity baseline:

```bash
-ctk q8_0 -ctv q8_0
```

Keep one paired default/F16-vs-Q8 check. Accept Q8 as production default if speed loss ≤3–5% and capacity materially rises. No Q4 KV unless Q8 remains capacity-bound.

### Automated model-agnostic tuning
1. **Default baseline:** no manual `--ctx-size`; retain unified KV. Parse chosen context/KV allocation. This tests llama.cpp default/fit behavior.
2. **Probe KV bytes/token empirically:** start known ctx pool; parse reported KV buffer bytes / cells. Handles dense, MoE, GQA/MQA, hybrid attention/conv automatically.
3. **Context fit search:** Q8 KV, safe ubatch 1024, candidate aggregate pools `128K→256K→512K→1M...`; geometric grow until startup/runtime failure. Choose 85–90% of max stable or largest needed candidate. No GB CLI required; translate via measured bytes/token.
4. **P search:** use durable full backlog, operational `Q=min(4P,32)`. Test P `1→2→4→8→16...`; each P gets enough requests to maintain backlog. Select min full-dataset wall with zero unrecovered failures. Stop after <5% throughput gain twice or capacity failure.
5. **ubatch:** at winning ctx/P, compare 1024→2048→4096; stop <5% gain/OOM/capacity loss.
6. **Retry validation:** deliberately pressure one level above winner; verify every failed ID requeues and completes at lower pressure.
7. **Alt-engine gate:** vLLM 0.21 smoke ≤5 min; SGLang FP16 smoke ≤5 min. Benchmark only if healthy.

Context and P are coupled. `--kv-unified` shares allocated cells but does not guarantee automatic use of all free VRAM. Omitting `--ctx-size` gives a required default baseline, not assumed max-capacity behavior. Worst-case P full-context guarantee needs `ctx_pool≈P×model_max_context`; otherwise multiplex empirically with durable retries.

## Required tuner fixes before spend
- Rename ambiguous `concurrency` reporting to client Q/effective C and server P.
- Keep P explicit/full command in every row.
- Serialize per-request failures/status/body, not count only.
- Preserve/retry failed prompt IDs durably.
- Parse llama.cpp KV/model/workspace allocation from logs.
- Report totals first: InputTPS, OutputTPS, wall, OK/fail/retry, failure action.
- Mark partial-success TPS invalid for optimization.
- Stream SSE for TTFT/ITL later; secondary for offline batch.
- Correct prompt selection: `--limit` currently takes head and can bias short prompts. Build tokenizer-exact deterministic mixed suites and shuffle deterministically.
- Global scheduler must chain config stages without chat intervention; reserve final 5 min/export.

## Persistence and locations
Everything created for this work is now under repository folder `ucloud_optimization/`:

```text
ucloud_optimization/
├── configs/
├── data/
├── docs/
├── scripts/
├── tools/
└── checkpoints/
```

Remote persistent UCloud folder remains `/work/LCPP_OffloadTesting` and contains models, SM100 build, CUDA runtime libs, envs, full logs. Compact local checkpoint: `ucloud_optimization/checkpoints/run-12372444/`.
