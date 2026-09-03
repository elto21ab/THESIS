#!/bin/bash
# Phase A provisioning: gemma-4-12B FP8 vs NVFP4 (job 12375721)
set -euo pipefail
BASE=/work/vLLM
mkdir -p $BASE/{venvs,models,data,experiments,results,tools,configs,flashinfer_cache,logs}
cp -an /work/hpc/data/V7_* $BASE/data/ 2>/dev/null || true
cp -an /work/hpc/tools/tune_inference.py /work/hpc/tools/prepare_benchmark_prompts.py $BASE/tools/ 2>/dev/null || true
cp -an /work/hpc/configs/inference-tune.example.toml $BASE/configs/ 2>/dev/null || true
ls $BASE/data | wc -l; ls $BASE/tools | wc -l

# flashinfer JIT cache: symlink home cache to persistent dir, seed from hpc
rm -rf ~/.cache/flashinfer
mkdir -p ~/.cache
cp -an /work/hpc/toolchain/flashinfer_cache/. $BASE/flashinfer_cache/ 2>/dev/null || true
ln -sfn $BASE/flashinfer_cache ~/.cache/flashinfer
ls $BASE/flashinfer_cache | wc -l

export CUDA_HOME=/work/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$CUDA_HOME/lib
export HF_HOME=$BASE/models
export HF_HUB_ENABLE_HF_TRANSFER=1

cd $BASE
echo "=== venv: stable vllm ==="
uv venv venvs/g4_stable --python 3.12
source venvs/g4_stable/bin/activate
uv pip install -U vllm "huggingface_hub[cli,hf_transfer]"
python -c "import vllm; print('VLLM_STABLE_VERSION', vllm.__version__)"
# arch support probe (gemma4 unified in stable?)
python - <<'EOF'
import glob, os, vllm
root = os.path.dirname(vllm.__file__)
hits = glob.glob(root + "/model_executor/models/gemma4*.py")
print("GEMMA4_MODULES", hits)
EOF
pip freeze > $BASE/venvs/g4_stable_freeze.txt

echo "=== downloads ==="
hf download RedHatAI/gemma-4-12B-it-FP8-Dynamic
hf download RedHatAI/gemma-4-12B-it-NVFP4
du -sh $BASE/models/hub/*
echo PROV_DONE
