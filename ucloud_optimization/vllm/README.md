# vLLM track — Q8 (FP8 W8A8) vs NVFP4 (W4A4)

**Objective:** accumulated ppTPS/tgTPS delta between FP8 W8A8 and NVFP4 W4A4 checkpoints in vLLM, minimal flags (concurrency/parallel only). Dense + MoE, multiple sizes.

**Result #1 (2026-08-29, Nemotron-Nano-9B-v2, mig1 slice): NVFP4 +11.9% ppTPS / +12.9% tgTPS / +53.6% KV pool. Zero failures.** → [experiments/2026-08-29_nemotron9b_q8_vs_nvfp4/README.md](experiments/2026-08-29_nemotron9b_q8_vs_nvfp4/README.md)

**Persistence rules (ephemeral /work root, mount gotchas, venv quirks, kill patterns): [runbook/PERSISTENCE_RULES.md](runbook/PERSISTENCE_RULES.md) — read first.**

**Remote root:** `/work/LCPP_OffloadTesting/vLLM/` (self-contained; toolchain shared w/ `hpc/toolchain` by path, no dup).
**Local mirror:** this folder.

## Status / next
- [ ] Folder reorg on remote (`/work/LCPP_OffloadTesting/vLLM/`), copy V7 data + tools from `hpc/`
- [ ] CPU provisioning job: nightly vLLM venv (Gemma-4 unified needs PR #44429, unshipped — stable 0.27 venv can't load it) + checkpoints (~25 GB) + flashinfer cache path
- [ ] 1 h MIG job (mig.1g.23gb, 1/7 GPU-h): smoke ×2 → serve-bench → results

## Experiment 1: gemma-4-12B-it, FP8 vs NVFP4
See [experiments/2026-XX_gemma4_q8_vs_nvfp4/PLAN.md](experiments/2026-XX_gemma4_q8_vs_nvfp4/PLAN.md) (to be created at run time).

### Checkpoints (verified 2026-XX-XX via HF API)
| repo | quant | repo size | fits mig.1g.23gb TP1 |
|---|---|---:|---|
| `RedHatAI/gemma-4-12B-it-FP8-Dynamic` | W8A8 (fp8 dynamic) | 15.0 GB | ✅ (~3.4 GB KV headroom @0.9 util) |
| `RedHatAI/gemma-4-12B-it-NVFP4` | W4A4 `nvfp4-pack-quantized` (compressed-tensors inline config, group 16, fp8 scales) | 10.3 GB | ✅ |
| `google/gemma-4-12B-it` (bf16 ref) | — | ~23.5 GB | ❌ TP1; TP2 = 2/7 billing |

## Settled facts
- Engine: vLLM ≥0.27 stable for non-Gemma4; **nightly cu129 wheel** for Gemma-4 unified (`vllm pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu129 --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match`).
- Gemma-4-12B arch: 48 layers = 40 SWA (window 1024, kv_heads 8, head_dim 256) + 8 full (global kv_heads 1, head_dim 512); 128K `max_position_embeddings` in config (card markets 256K — don't raise).
- **fp8 KV sizing: ~8 KiB/token full-layers + 160 MiB/seq SWA ceiling → 1M ctx ≈ 8.4 GB fp8 (16.8 GB bf16).** Verify vs startup `GPU KV cache size` log line.
- MIG rules: same-slice A/B valid; absolute TPS ≈ 1/7; kernel/BW/TP-comm wins may flip on full GPU → validate winners once on full B200. Symm-mem off (`VLLM_ALLREDUCE_USE_SYMM_MEM=0`).
- TP = capacity knob (shards weights+KV, handles any quant > 1 GPU; needs TP-strategy job). DP = throughput knob (TP1 × N replicas / `--data-parallel-size` beats TP8 when quant fits 1 GPU). MoE: `--enable-expert-parallel` option.

## Layout (mirror of remote)
```
vllm/
├── runbook/         SSH + prechecklist (adapted from hpc)
├── experiments/     one dir per experiment w/ README
├── tools/           tune_inference.py (vLLM serve-bench capable), prepare_benchmark_prompts.py
├── configs/         inference-tune tomls
├── data/            V7 suites + manifests
├── results/         compact result sheets
└── checkpoints/     small artifacts
```

## Docs (start here)
1. [runbook/UCLOUD_SSH_RUNBOOK.md](runbook/UCLOUD_SSH_RUNBOOK.md)
2. [runbook/UCLOUD_VLLM_PRECHECKLIST.md](runbook/UCLOUD_VLLM_PRECHECKLIST.md)
