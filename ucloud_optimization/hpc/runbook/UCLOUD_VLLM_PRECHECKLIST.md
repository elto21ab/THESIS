# UCloud vLLM Pre-flight Checklist (MIG / Blackwell)

> **2026-08-29 remote restructure:** everything under `/work/LCPP_OffloadTesting` now lives in `hpc/` subdirs. Paths below already reflect the NEW layout. `llama-inference/` was folded into `hpc/experiments/2026-08-19_llama_first/llama-inference` and its source dir emptied.

Run these in order at job start. Each item is a real past failure. Do not skip.

## 1. Hardware → TP math (before submitting the job)
- Model weights/rank = params × bytes/param × (1 + overhead≈5–10%).
- Slice VRAM on `mig.1g.23gb` = 20.5 GB usable.
- If weights/rank > slice VRAM → TP2 mandatory (bills 2/7 GPU-h/h). TP1 arm = dead (e.g. 27B NVFP4 ≈ 24 GiB/rank).
- 8×B200 node = 8 independent GPUs. Dense 27B → TP1 × 8 replicas beats TP8.

## 2. Toolchain restore (SSH in, before any serve)
```bash
export CUDA_HOME=/work/LCPP_OffloadTesting/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```
Verify: `nvcc --version`; headers present: `cuda_runtime.h cublasLt.h nvrtc.h curand.h` in `$CUDA_HOME/include`; `lib64` → `lib` symlink; `stubs/libcuda.so`.
Toolchain is fully self-contained (cublas/cublasLt merged in 2026-08-29) — no apt step needed. Backup debs in `/work/LCPP_OffloadTesting/hpc/toolchain/pkg/` (1.4 GB, 14 debs incl. cuda-keyring recipe: `dpkg -i cuda-keyring_*.deb && apt-get update && apt-get download <pkgs>`; note `cuda-cub`/`cuda-thrust` are virtual stubs — headers live in `cuda-cccl-13-0`).

## 3. FlashInfer JIT cache (biggest time sink if forgotten)
Cache dir `~/.cache/flashinfer` is ephemeral per job (~9 min rebuild). Persist it:
```bash
mkdir -p /work/LCPP_OffloadTesting/hpc/toolchain/flashinfer_cache
cp -a ~/.cache/flashinfer/* /work/LCPP_OffloadTesting/hpc/toolchain/flashinfer_cache/ 2>/dev/null
export FLASHINFER_CACHE_DIR=/work/LCPP_OffloadTesting/hpc/toolchain/flashinfer_cache   # if env var honored by this flashinfer version; else symlink ~/.cache/flashinfer -> it
```
First serve on a fresh cache JIT-compiles sampling + fmha_gen etc. — expected, takes minutes. Subsequent runs hit cache.

## 4. Env block for EVERY vllm serve/bench invocation
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # MIG NVML assert; capital F
export CUDA_HOME=/work/LCPP_OffloadTesting/hpc/toolchain/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export VLLM_ALLREDUCE_USE_SYMM_MEM=0                        # symm-mem fails on MIG; use PYNCCL
```

## 5. Smoke serve → health → kill → bench
```bash
cd /work/LCPP_OffloadTesting/hpc && source venvs/qwen38_vllm/bin/activate
setsid nohup vllm serve <MODEL> --port 8000 --tensor-parallel-size 2 > /tmp/serve.log 2>&1 < /dev/null &
# poll: curl -sf localhost:8000/health  →  then pkill (separate ssh call, bracket pattern)
```
Minimal flags only (user pref): port + TP. No `--max-model-len`. Engine manages parallelism.

## 6. pkill / session hygiene (all burned in)
- Never `pkill -f "vllm serve"` in the same ssh compound command that contains the literal text — it matches its own cmdline. Use `pkill -f "[v]llm serve"` in a **separate** call.
- Remote tmux dies with SSH (no linger) → `setsid nohup ... &` + log tail instead.
- Verify every `cp`/`mv` (`ls | wc -l`), never swallow stderr during provisioning.
- `export A=x ... $A` in one command expands before assignment → split statements.
- apt: one bad package name aborts the whole transaction. Install dev packages individually or verify each name first.

## 7. UCloud UI specifics
- Machine: `gpu-nvidia-b200-2-mig.1g`, STRATEGY=Tensor Parallel A, KV OFFLOAD=Off.
- SSH port is per-job, shown on job page → `ssh ucloud@ssh.cloud.sdu.dk -p <port>`.
- "Stop application" is **hold-to-confirm** in current UI (first click may just focus) — use `pb hold <sess> <ref> --ms 2500` (added to patchright skill 2026-08-29).
- Files UI (`app/files?path=/12370687/...`) is read-only browse; moving/renaming must be done in a job terminal via SSH.
- Cheap filesystem-only ops: submit `terminal-ubuntu` on `cpu-amd-zen5-1-vcpu` (1 core-hour) w/ folders attached; note "Add folder" requires clicking the folder row in the picker dialog — an empty-looking row means it didn't attach.

## Quant lessons (settled)
- **Engine settled: vLLM ≥ 0.27** for NVFP4 on Blackwell. SGLang only if it beats vLLM materially.
- NVFP4 needs sm100/sm120 (Blackwell). Kernels confirmed on MIG SM100: `CutlassNvFp4LinearKernel`, `FlashInferCuteDslNvFp4LinearKernel`.
- GGUF ↔ vLLM: GGUF is llama.cpp's format; vLLM/SGLang use HF safetensors + `hf_quant_config.json`. Do not mix.
- heretic (uncensoring) needs bf16 weights; run it before quantization, not on a NVFP4 checkpoint.
- fp8 on-the-fly ≈ +25% tg vs bf16 on B200 (tensor cores). NVFP4 expected ~2× tg but checkpoint must fit + load cleanly.
- Offload ANY % of weights kills vLLM throughput (layer-granular prefetch, not router-aware) → avoid; llama.cpp `-ngl` is router-aware for the too-big case.

## Where things live (persistent, `/work/LCPP_OffloadTesting/hpc/`)
| path | content |
|---|---|
| `venvs/qwen38_vllm` | vLLM ≥0.27 (see `experiments/2026-08-28_qwen38_nvfp4/requirements_freeze.txt`) |
| `models/qwen38_hf_cache` | both Qwen3.8-27B-NVFP4 checkpoints (HF cache layout) |
| `models/` | LFM2.5 / Gemma-4 / Nemotron checkpoints from earlier sweeps |
| `toolchain/pkg` | 1.4 GB CUDA dev debs (complete set) |
| `toolchain/cuda-13` | full sm100 toolchain (incl. cublasLt, stubs, lib64 symlink) |
| `toolchain/flashinfer_cache` | JIT cache (populate on next GPU run) |
| `data/` | V7 prompt suites + batches |
| `venvs/{vllm,sglang,tuner}` | historical venvs from 0826 sweep |
| `results/runs_archive` | old 0826 benchmark JSONs |
| `experiments/` | per-date experiment artifacts + logs |
