# Experiment 1: gemma-4-12B-it — FP8 (W8A8) vs NVFP4 (W4A4)

**Q:** accumulated ppTPS/tgTPS delta between the two quants, minimal flags (concurrency/parallel only) in vLLM.

**Protocol (revised):** NO concurrency ladder. One pass per arm: all 48 prompts in flight (Q=48), no `--max-num-seqs` cap — vLLM continuous batching auto-fills the GPU. Minimal serve flags: `--port` + text-only MM-disable (workload-defining). Client: temp 0, max_tokens 256, non-streaming.

## Arms
| arm | repo | quant | GB |
|---|---|---|---:|
| Q8 | `RedHatAI/gemma-4-12B-it-FP8-Dynamic` | fp8 W8A8 dynamic | 15.0 |
| NVFP4 | `RedHatAI/gemma-4-12B-it-NVFP4` | nvfp4 W4A4 (compressed-tensors inline) | 10.3 |

Same producer/recipe (LLM Compressor) → clean A/B. Both TP1 on mig.1g.23gb → 1/7 GPU-h billing.

**FP8 size sanity (15 GB ≠ 2× NVFP4, ≠ 12 GB naive):** 11.95B params — linear layers ≈ 10.9B × 1 B (E4M3) ≈ 10.9 GB; embeddings (vocab 262144 × 3840 ≈ 1.0B) + mm towers kept bf16 ≈ +2.1 GB; per-channel scales + safetensors overhead ≈ +1.5 GB → ~15 GB. bf16 ref 23.5 GB; ratio 0.64 not 0.5, expected for fp8-dynamic w/ bf16 outliers kept.

**Arch support ≠ quant support:** FP8-Dynamic loads on stable vLLM (recipe: 0.25.0+ applies to gemma-4 quant variants generally; 26B-A4B recipe stable-supported). BUT 12B *encoder-free unified* arch only in nightly (PR #44429) — nightly venv still required for THIS model regardless of quant. Nightly-avoid alternative pair: `RedHatAI/gemma-4-26B-A4B-it-{FP8-Dynamic,NVFP4}` (stable 0.25+), but FP8 26B ≈ 27 GB > MIG slice → needs full B200 (TP1, trivial fit) or MIG TP2 (2/7). Keep 12B for MIG screening; graduate 26B-A4B pair to full B200 when committing models.

## Phase A — CPU provisioning job (terminal-ubuntu, 1 vcpu, ~0 cost)
```bash
# remote
mkdir -p /work/LCPP_OffloadTesting/vLLM/{venvs,models,data,experiments,results,flashinfer_cache}
cp -a /work/LCPP_OffloadTesting/hpc/data/V7_* /work/LCPP_OffloadTesting/vLLM/data/
cp -a /work/LCPP_OffloadTesting/hpc/tools/{tune_inference.py,prepare_benchmark_prompts.py} /work/LCPP_OffloadTesting/vLLM/tools/
# nightly venv (Gemma-4 unified arch NOT in stable; PR #44429)
uv venv /work/LCPP_OffloadTesting/vLLM/venvs/gemma4_nightly --python 3.12
source /work/LCPP_OffloadTesting/vLLM/venvs/gemma4_nightly/bin/activate
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
pip freeze > /work/LCPP_OffloadTesting/vLLM/venvs/gemma4_nightly_freeze.txt
# checkpoints
HF_HOME=/work/LCPP_OffloadTesting/vLLM/models hf download RedHatAI/gemma-4-12B-it-FP8-Dynamic
HF_HOME=/work/LCPP_OffloadTesting/vLLM/models hf download RedHatAI/gemma-4-12B-it-NVFP4
```
Verify: `ls` counts both repos (~15 GB, ~10.3 GB); `python -c "import vllm"`.

## Phase B — GPU job (1× mig.1g.23gb, 1 h, hard budget)
Env (every invocation):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_HOME=/work/LCPP_OffloadTesting/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export FLASHINFER_CACHE_DIR=/work/LCPP_OffloadTesting/vLLM/flashinfer_cache
export HF_HOME=/work/LCPP_OffloadTesting/vLLM/models
```
1. Smoke FP8: `vllm serve RedHatAI/gemma-4-12B-it-FP8-Dynamic --port 8000 --limit-mm-per-prompt '{"image":0,"audio":0}'` → `/health` → kill. First serve = flashinfer JIT ~9 min (cache persisted).
2. Smoke NVFP4: same, port 8001. Confirm kernel path `CutlassNvFp4LinearKernel` or `FlashInferCuteDslNvFp4LinearKernel` in log; record `GPU KV cache size` line from both.
3. Bench: `tune_inference.py` serve-bench (vLLM engine path), suite = `V7_perf_real_48.jsonl` (48 real mixed-length prompts, 3k–132k chars, ~542k pp tokens total, deterministic SHA'd). Q=48, single pass per arm. Budget ~6–10 min/pass on MIG; if behind at T+30 min, cut to 24-prompt head subset (deterministic, note deviation).
4. Copy `~/.cache/flashinfer/*` → `vLLM/flashinfer_cache/` before job end. Export JSONL + README to `vLLM/experiments/2026-XX_gemma4_q8_vs_nvfp4/`.

## Deliverable
| Engine/Model/Quant/GB | P | Q | ctx pool | KV type | ppTPS | tgTPS | wall | OK | failed | failure behavior | full flags |
|---|---|---|---|---|---|---|---|---|---|---|---|

Δ% FP8→NVFP4 per C + pooled. Mark partial-success passes invalid.

## Risks
- Nightly wheel × sm100/MIG combo untested → fallback: Qwen3.8 FP8?/NVFP4 pair on stable 0.27 venv, or 26B-A4B pair on full B200.
- 1 h tight: JIT 9 min + 2 smoke + 2 long passes (~20 min) + reserve. T+30 checkpoint rule above.
- FP8 arm KV headroom ~3.4 GB @0.9 util → ~400k tok fp8 KV; suite max seq ~36k tok fits single-seq but long prompts × 48 concurrent ≠ all resident — vLLM schedules what fits (prefix-cache recompute allowed); zero-failure expected, log `GPU KV cache size` line.
