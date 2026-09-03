#!/bin/bash
# Nemotron-Nano-9B-v2 FP8 vs NVFP4 — v5: ALL paths under /work/hpc (persistent mount; /work root = ephemeral!)
set -uo pipefail
BASE=/work/hpc/vLLM
VENV=/work/hpc/venvs/qwen38_vllm
EXP=$BASE/experiments/2026-08-29_nemotron9b_q8_vs_nvfp4
mkdir -p $EXP/logs $BASE/flashinfer_cache
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a $EXP/timeline.log; }
mark "START nemotron9b v5"

export PATH=$VENV/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_HOME=/work/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HF_HOME=$BASE/models

# persist flashinfer JIT cache (symlink home cache into /work/hpc)
rm -rf ~/.cache/flashinfer
ln -sfn $BASE/flashinfer_cache ~/.cache/flashinfer

FP8=nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8
NV4=nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4

mark "downloads start"
python - <<'EOF'
from huggingface_hub import snapshot_download
for rid in ["nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8", "nvidia/NVIDIA-Nemotron-Nano-9B-v2-NVFP4"]:
    p = snapshot_download(rid)
    print("DOWNLOADED", rid, p, flush=True)
EOF
mark "downloads done"

python - <<'EOF'
import json
rows=[json.loads(l) for l in open('/work/hpc/vLLM/data/V7_perf_real_48.jsonl')]
rows=sorted(rows,key=lambda r:r.get('chars',0))
sub=[rows[round(i*(len(rows)-1)/11)] for i in range(12)]
sub=sorted(sub,key=lambda r:r.get('chars',0))
with open('/work/hpc/vLLM/data/V7_real_12.jsonl','w') as f:
    for r in sub: f.write(json.dumps(r)+'\n')
print('SUBSET12_CHARS', sum(r.get('chars',0) for r in sub))
EOF
REAL12=$BASE/data/V7_real_12.jsonl
CLI="$VENV/bin/python $BASE/tools/bench_client.py"

serve() { setsid nohup python -m vllm serve "$1" --port "$2" > "$3" 2>&1 < /dev/null & }
wait_health() { for i in $(seq 1 150); do curl -sf localhost:$1/health >/dev/null && return 0; sleep 4; done; return 1; }
kill_serve() { pkill -f "[v]llm serve" 2>/dev/null; sleep 12; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 5; }
kvline() { grep -E "GPU KV cache size|Maximum concurrency for|Available KV cache memory" "$1" | head -4; }

run_pass() { mark "pass $7"; $CLI http://localhost:$1 "$2" "$3" "$4" "$5" "$6" "$7" || true; }

run_arm() { # model port tag
  local model=$1 port=$2 tag=$3
  mark "serve $tag start"
  serve "$model" $port $EXP/logs/serve_$tag.log
  if wait_health $port; then
    mark "health OK $tag"
    kvline $EXP/logs/serve_$tag.log > $EXP/kv_$tag.txt
    run_pass $port "$model" "$REAL12" $EXP/${tag}_full.jsonl 256 12 all   ${tag}_full
    run_pass $port "$model" "$REAL12" $EXP/${tag}_pp.jsonl   1   12 all   ${tag}_pp
    run_pass $port "$model" "$REAL12" $EXP/${tag}_tg.jsonl   256 1  median ${tag}_tg
  else
    mark "HEALTH TIMEOUT $tag"
  fi
  kill_serve
}

T0=$(date +%s)
run_arm $FP8 8000 fp8
run_arm $NV4 8001 nvfp4
nvidia-smi > $EXP/logs/nvidia_smi_final.txt 2>&1
pip freeze > $EXP/requirements_freeze.txt
mark "PHASE_B_DONE elapsed=$(( $(date +%s) - T0 ))s"
