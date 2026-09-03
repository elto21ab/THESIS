#!/bin/bash
# NVFP4-only arm (FP8 done 2026-08-29). Fixed kill pattern: shim cmdline = "python -c from vllm.entrypoints..."
set -uo pipefail
BASE=/work/hpc/vLLM
VENV=/work/hpc/venvs/qwen38_vllm
EXP=$BASE/experiments/2026-08-29_nemotron9b_q8_vs_nvfp4
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a $EXP/timeline.log; }
mark "START nvfp4 arm"

export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_HOME=/work/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HF_HOME=$BASE/models

# persist torch.compile cache (AOT artifacts are per-arch; reuse across arms)
rm -rf ~/.cache/vllm; ln -sfn $BASE/torch_compile_cache ~/.cache/vllm; mkdir -p $BASE/torch_compile_cache

NV4=nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4
REAL12=$BASE/data/V7_real_12.jsonl
CLI="$VENV/bin/python $BASE/tools/bench_client.py"

mark "serve nvfp4 start"
setsid nohup vllm serve $NV4 --port 8001 > $EXP/logs/serve_nvfp4_retry.log 2>&1 < /dev/null &
for i in $(seq 1 150); do curl -sf localhost:8001/health >/dev/null && break; sleep 4; done
if curl -sf localhost:8001/health >/dev/null; then
  mark "health OK nvfp4"
  grep -E "Available KV cache memory|GPU KV cache size" $EXP/logs/serve_nvfp4_retry.log | head -3 > $EXP/kv_nvfp4.txt
  mark "pass nvfp4_full";  $CLI http://localhost:8001 $NV4 $REAL12 $EXP/nvfp4_full.jsonl 256 12 all   nvfp4_full || true
  mark "pass nvfp4_pp";    $CLI http://localhost:8001 $NV4 $REAL12 $EXP/nvfp4_pp.jsonl   1   12 all   nvfp4_pp || true
  mark "pass nvfp4_tg";    $CLI http://localhost:8001 $NV4 $REAL12 $EXP/nvfp4_tg.jsonl   256 1  median nvfp4_tg || true
else
  mark "HEALTH TIMEOUT nvfp4"
fi
pkill -f "entrypoints.cli.main"; sleep 8; pkill -9 -f "entrypoints.cli.main" 2>/dev/null
mark "NVFP4_ARM_DONE"
