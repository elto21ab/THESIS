#!/bin/bash
# Phase B v2: gemma-4-12B FP8 vs NVFP4 — uniform A/B @ util 0.97
# Per arm × per suite: full (Q=48, 256 tok) + pp-only (Q=48, max_tokens=1) + tg single-stream (Q=1 median, 256 tok)
set -uo pipefail
BASE=/work/vLLM
EXP=$BASE/experiments/2026-08-29_gemma4_q8_vs_nvfp4
mkdir -p $EXP/logs
mark() { echo "[$(date +%H:%M:%S)] $*" | tee -a $EXP/timeline.log; }
mark "START phase B v2"

source $BASE/venvs/g4_stable/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_HOME=/work/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HF_HOME=$BASE/models

FP8=RedHatAI/gemma-4-12B-it-FP8-Dynamic
NV4=RedHatAI/gemma-4-12B-it-NVFP4
REAL=$BASE/data/V7_perf_real_48.jsonl
REAS=$BASE/data/V7_perf_reasoning_48.jsonl
# deterministic stratified 24/48 subset (sorted by chars, evenly spaced) — 1h job budget
python - <<'EOF'
import json
rows=[json.loads(l) for l in open('/work/vLLM/data/V7_perf_real_48.jsonl')]
rows=sorted(rows,key=lambda r:r.get('chars',0))
sub=[rows[round(i*(len(rows)-1)/23)] for i in range(24)]
sub=sorted(sub,key=lambda r:r.get('chars',0))
with open('/work/vLLM/data/V7_real_24.jsonl','w') as f:
    for r in sub: f.write(json.dumps(r)+'\n')
print('SUBSET24_CHARS', sum(r.get('chars',0) for r in sub))
EOF
REAL24=$BASE/data/V7_real_24.jsonl
CLI="python $BASE/tools/bench_client.py"

# deviations (documented): util 0.92 (0.97 = profiling OOM/NVML assert on MIG; 0.9 boots) + max-model-len 131072 (recipe pin; MIG slice can't host 262144-ctx KV for FP8: 5.84 GiB needed > pool)
serve() { setsid nohup vllm serve "$1" --port "$2" --gpu-memory-utilization 0.92 --max-model-len 131072 --limit-mm-per-prompt '{"image":0,"audio":0}' > "$3" 2>&1 < /dev/null & }
wait_health() { for i in $(seq 1 150); do curl -sf localhost:$1/health >/dev/null && return 0; sleep 4; done; return 1; }
kill_serve() { pkill -f "[v]llm serve" 2>/dev/null; sleep 12; pkill -9 -f "[v]llm serve" 2>/dev/null; sleep 5; }
kvline() { grep -E "GPU KV cache size|Maximum concurrency for|Available KV cache memory" "$1" | head -4; }

run_pass() { # port model suite out maxtok q select tag
  mark "pass $tag"
  $CLI http://localhost:$1 "$2" "$3" "$4" "$5" "$6" "$7" || true
}

run_arm() { # model port tag suite suite_tag
  local model=$1 port=$2 tag=$3 suite=$4 stag=$5
  mark "serve $tag/$stag start"
  serve "$model" $port $EXP/logs/serve_${tag}_${stag}.log
  if wait_health $port; then
    mark "health OK $tag/$stag"
    kvline $EXP/logs/serve_${tag}_${stag}.log > $EXP/kv_${tag}_${stag}.txt
    run_pass $port "$model" "$suite" $EXP/${tag}_${stag}_full.jsonl 256 24 all  ${tag}_${stag}_full
    run_pass $port "$model" "$suite" $EXP/${tag}_${stag}_pp.jsonl   1   24 all  ${tag}_${stag}_pp
    run_pass $port "$model" "$suite" $EXP/${tag}_${stag}_tg.jsonl   256 1  median ${tag}_${stag}_tg
  else
    mark "HEALTH TIMEOUT $tag/$stag"
  fi
  kill_serve
}

T0=$(date +%s)
run_arm $FP8 8000 fp8 $REAL24 real
run_arm $NV4 8001 nvfp4 $REAL24 real
nvidia-smi > $EXP/logs/nvidia_smi_final.txt 2>&1
pip freeze > $EXP/requirements_freeze.txt
mark "PHASE_B_DONE elapsed=$(( $(date +%s) - T0 ))s"
