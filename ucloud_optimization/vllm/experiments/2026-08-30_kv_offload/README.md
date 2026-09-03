# KV-offload variants — Nemotron-Nano-9B-v2-NVFP4, mig.1g.23gb, vLLM 0.28 (2026-08-30)

Recipe variants: https://recipes.vllm.ai/nvidia/NVIDIA-Nemotron-Nano-9B-v2?variant=nvfp4 (KV Offload: Off / Simple / Mooncake / LMCache).

## Results (V7_real_12, 12 reqs, conc 64→Q12, max 64 tok)

| arm | flags | KV pool | ppTPS | tgTPS | lat p50 | verdict |
|---|---|---:|---:|---:|---:|---|
| off (baseline) | `--gpu-memory-utilization 0.95` | 10.71 GiB / 614k tok | **11280** | **63.8** | 8.76s | winner |
| cpuoffload | `--kv-transfer-config OffloadingConnector, cpu_bytes 8GB` | 9.05 GiB / 520k tok | 10763 (−4.6%) | 60.9 (−4.6%) | 9.32s | slower, −15% GPU pool |
| lmcache | `LMCacheConnectorV1` | — | — | — | — | **crashed**: "Failed to promote local KV cache specs to one unified type" |
| mooncake | — | — | — | — | — | skipped: needs mooncake store daemon, multi-node only |

## Interpretation

- **OffloadingConnector reserves GPU KV buffer space** (`kv_buffer_device=cuda`, 1 GiB default buffer) + pays CPU↔GPU transfer → strictly worse on short-ctx Q12 bench. Win case = ctx > GPU pool (614k tok here — suite never gets close). MIG slice has 2266 GB RAM host-side, so offload only matters at extreme concurrency/ctx.
- **LMCache broken on this model**: Mamba-hybrid → two KV-cache spec types (mamba state + full attn), LMCache connector requires unified spec → ValueError at engine init. Also `lmcache` pkg not in venv (would be next failure). Verdict: **no KV-connector support for Mamba-hybrid models in vLLM 0.28** — matches spec-dec limitation. Nemotron-Nano = plain serving only.
- **Mooncake**: needs external store + RDMA; single-slice test pointless.

**Settled:** NVFP4-on-MIG workflow = plain serve, mem-util 0.95. Revisit offloading on full B200 w/ 131k-ctx suite where GPU pool binds.

## Files
`kvoffload.sh` (runner), `timeline.log`, `logs/serve_{off,cpuoffload,lmcache}.log`, `{off,cpuoffload}_full.jsonl`, `kv_*.txt` (KV pool lines).
