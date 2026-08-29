# UCloud vLLM Pre-flight Checklist (MIG / Blackwell)

Run these in order at job start. Each item is a real past failure. Do not skip.

## 1. Hardware → TP math (before submitting the job)
- Model weights/rank = params × bytes/param × (1 + overhead≈5–10%).
- Slice VRAM on `mig.1g.23gb` = 20.5 GB usable.
- If weights/rank > slice VRAM → TP2 mandatory (bills 2/7 GPU-h/h). TP1 arm = dead (e.g. 27B NVFP4 ≈ 24 GiB/rank).
- 8×B200 node = 8 independent GPUs. Dense 27B → TP1 × 8 replicas beats TP8.

## 2. Toolchain restore (SSH in, before any serve)
```bash
export CUDA_HOME=/work/LCPP_OffloadTesting/runtime/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```
Verify: `nvcc --version`; headers present: `cuda_runtime.h cublasLt.h nvrtc.h curand.h` in `$CUDA_HOME/include`; `lib64` → `lib` symlink; `stubs/libcuda.so`.
If missing: re-install debs from `/work/LCPP_OffloadTesting/qwen38/pkg/` (519 MB, all deps local):
```bash
sudo dpkg -i /work/LCPP_OffloadTesting/qwen38/pkg/*.deb
# then re-copy include/lib into runtime/cuda-13 if needed (source: /usr/local/cuda-13.0/{include,lib64,bin})
```

## 3. FlashInfer JIT cache (biggest time sink if forgotten)
Cache dir `~/.cache/flashinfer` is ephemeral per job (~9 min rebuild). Persist it:
```bash
mkdir -p /work/LCPP_OffloadTesting/flashinfer_cache
cp -a ~/.cache/flashinfer/* /work/LCPP_OffloadTesting/flashinfer_cache/ 2>/dev/null
export FLASHINFER_CACHE_DIR=/work/LCPP_OffloadTesting/flashinfer_cache   # if env var honored by this flashinfer version; else symlink ~/.cache/flashinfer -> it
```
First serve on a fresh cache JIT-compiles sampling + fmha_gen etc. — expected, takes minutes. Subsequent runs hit cache.

## 4. Env block for EVERY vllm serve/bench invocation
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False   # MIG NVML assert; capital F
export CUDA_HOME=/work/LCPP_OffloadTesting/runtime/cuda-13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export VLLM_ALLREDUCE_USE_SYMM_MEM=0                        # symm-mem fails on MIG; use PYNCCL
```

## 5. Smoke serve → health → kill → bench
```bash
cd /work/LCPP_OffloadTesting/qwen38 && source venv/bin/activate
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
- "Stop application" raises native `confirm()` → in patchright: `pb click <sess> <ref> --accept-dialog`.
- Files UI (`app/files?path=/12370687/...`) is read-only browse; moving/renaming must be done in a job terminal via SSH.

## Quant lessons (settled)
- **Engine settled: vLLM ≥ 0.27** for NVFP4 on Blackwell. SGLang only if it beats vLLM materially.
- NVFP4 needs sm100/sm120 (Blackwell). Kernels confirmed on MIG SM100: `CutlassNvFp4LinearKernel`, `FlashInferCuteDslNvFp4LinearKernel`.
- GGUF ↔ vLLM: GGUF is llama.cpp's format; vLLM/SGLang use HF safetensors + `hf_quant_config.json`. Do not mix.
- heretic (uncensoring) needs bf16 weights; run it before quantization, not on a NVFP4 checkpoint.
- fp8 on-the-fly ≈ +25% tg vs bf16 on B200 (tensor cores). NVFP4 expected ~2× tg but checkpoint must fit + load cleanly.
- Offload ANY % of weights kills vLLM throughput (layer-granular prefetch, not router-aware) → avoid; llama.cpp `-ngl` is router-aware for the too-big case.

## Where things live (persistent, `/work/LCPP_OffloadTesting/`)
| path | content |
|---|---|
| `qwen38/venv` | vLLM ≥0.27 (see `requirements_freeze.txt`) |
| `qwen38/hf` | both Qwen3.8-27B-NVFP4 checkpoints |
| `qwen38/pkg` | 519 MB CUDA dev debs |
| `qwen38/{V7_perf_real_48,V7_perf_reasoning_48}.jsonl` | bench prompts |
| `runtime/cuda-13` | full sm100 toolchain |
| `flashinfer_cache` | JIT cache (create + populate on next run) |
| `venvs/` | historical venvs from 0826 sweep |
| `runs/`, `results/` | old 0826 benchmark JSONs |
