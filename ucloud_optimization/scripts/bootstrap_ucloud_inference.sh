#!/usr/bin/env bash
# Prepare a UCloud Ubuntu Terminal B200 job. Safe to rerun.
set -Eeuo pipefail
ROOT=${ROOT:-/work/LCPP_OffloadTesting}
JOBS=${JOBS:-6}
CUDA_VER=${CUDA_VER:-13-0}
mkdir -p "$ROOT"/{models,runs,src,venvs}
cd "$ROOT"

# Hard fail if UCloud did not attach an explicit folder resource. Container /work can vanish.
if [[ -f /work/JobParameters.json ]] && ! grep -q '"resources":\[[^]]' /work/JobParameters.json; then
  echo 'ERROR: no persistent folder resource in JobParameters.json. Stop job; mount LCPP_OffloadTesting correctly.' >&2
  exit 2
fi

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends ca-certificates wget git cmake ninja-build build-essential python3-venv sysstat numactl jq
if [[ ! -x /usr/local/cuda-${CUDA_VER/-/.}/bin/nvcc ]]; then
  cd /tmp
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    "cuda-compiler-${CUDA_VER}" "cuda-cudart-dev-${CUDA_VER}" "libcublas-dev-${CUDA_VER}"
fi
export PATH="/usr/local/cuda-${CUDA_VER/-/.}/bin:$PATH"

if [[ ! -d src/llama.cpp/.git ]]; then git clone --depth 1 https://github.com/ggml-org/llama.cpp src/llama.cpp; fi
cmake -S src/llama.cpp -B src/llama.cpp/build -G Ninja \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100 -DCMAKE_BUILD_TYPE=Release
cmake --build src/llama.cpp/build -j"$JOBS"

{
  date -Is
  nvidia-smi -L
  /usr/local/cuda-${CUDA_VER/-/.}/bin/nvcc --version
  src/llama.cpp/build/bin/llama-server --version
  cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
  cat /sys/fs/cgroup/memory.max 2>/dev/null || true
} | tee environment.txt

echo "READY: $ROOT"
