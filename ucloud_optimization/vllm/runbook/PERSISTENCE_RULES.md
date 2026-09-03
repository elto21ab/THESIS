# Persistence rules (UCloud /work) — READ BEFORE ANY JOB

- `/work` root = EPHEMERAL per-job disk. Anything written outside an attached folder dies with the job.
- Attached folder `/12370687/LCPP_OffloadTesting/hpc` mounts at `/work/hpc` (basename). Everything persistent lives under it:
  - `/work/hpc/vLLM/{models,venv-shims,experiments,data,flashinfer_cache,torch_compile_cache}` — vLLM track
  - `/work/hpc/toolchain/cuda-13` — sm100 toolchain; `/work/hpc/venvs/qwen38_vllm` — vLLM 0.28.0 venv
- Symlink ephemeral caches into persistent dirs at job start:
  - `ln -sfn /work/hpc/vLLM/flashinfer_cache ~/.cache/flashinfer`
  - `ln -sfn /work/hpc/vLLM/torch_compile_cache ~/.cache/vllm`
- uv venv gotchas: `source activate` does NOT change PATH → use `$VENV/bin/python` + `export PATH=$VENV/bin:$PATH`.
  No console scripts → shim at `/work/hpc/venvs/qwen38_vllm/bin/vllm` (recreate: `exec $V/bin/python -c "from vllm.entrypoints.cli.main import main; import sys; sys.exit(main())" "$@""`). `python -m vllm` does NOT work on 0.28.
- Server kill: cmdline = `python -c from vllm.entrypoints.cli.main...` → `pkill -f "entrypoints.cli.main"`, then sleep 8 + `pkill -9`, then verify VRAM released before next serve.
- pkill foot-gun: never combine a kill pattern with any other text matching it in the same command line (incl. paths) — write ops to a script, scp, bash it.
- UCloud UI: folder attach requires selecting the dropdown row (typing alone silently attaches nothing). Robust: "Import parameters" from known-good job. Stop = hold-to-confirm (`pb hold <sess> <ref> --ms 2500`).
- Watchers: ssh-only, poll `sleep 60`, scope greps to newest timeline segment (`tac file | awk "/START {exit} {print}" | tac`) to avoid stale-marker false fires.
