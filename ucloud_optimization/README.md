# UCloud inference optimization

**Tracks:**
- `hpc/` — llama.cpp primary + engine sweeps (historical, settled: vLLM ≥0.27 for NVFP4 on Blackwell).
- [`vllm/`](vllm/README.md) — **current track**: vLLM-only, Q8 (FP8 W8A8) vs NVFP4 (W4A4) benchmarking across model sizes. Remote: `/work/LCPP_OffloadTesting/vLLM/`.
- [`donations/`](donations/README.md) — participant data-donation pipeline (clean-bundle uploads into UCloud). Code done + proven via tunnel; **blocked on SDU public-URL provisioning** — see README + API notes.

**Start here (order, hpc track):**
1. [`hpc/runbook/UCLOUD_SSH_RUNBOOK.md`](hpc/runbook/UCLOUD_SSH_RUNBOOK.md) — how to SSH, billing, job UI quirks, pkill foot-guns.
2. [`hpc/runbook/UCLOUD_VLLM_PRECHECKLIST.md`](hpc/runbook/UCLOUD_VLLM_PRECHECKLIST.md) — pre-flight checklist, env block, quant lessons. **Read before launching any vLLM job.**
3. [`hpc/experiments/README.md`](hpc/experiments/README.md) — experiment index (what was tried, verdicts, links).
4. [`hpc/runbook/DECISIONS_AND_NEXT_SESSION.md`](hpc/runbook/DECISIONS_AND_NEXT_SESSION.md) — current objective (aggregate pp/tg TPS per billed GPU-hour), metric definitions, next steps.
5. [`hpc/runbook/INFERENCE_TUNING_HANDOFF.md`](hpc/runbook/INFERENCE_TUNING_HANDOFF.md) — full historical handoff (older, superseded in parts by the prechecklist).

## Layout
```
hpc/
├── runbook/         operational docs (SSH, prechecklist, decisions, historical handoff)
├── experiments/     one dir per experiment, each with README (postmortem + results)
├── results/         compact validated result sheets (RESULTS_8B.md, ALT_ENGINES.md)
├── tools/           tune_inference.py (serve-bench driver), prepare_benchmark_prompts.py
├── configs/         inference-tune.example.toml
├── data/            V7 benchmark prompt suites (+ manifests)
├── scripts/         UCloud bootstrap + upload + provisioning
└── checkpoints/     tarballs + small artifacts (incl. run-12372444 final run, qwen38 0828)
```

## Policy
No token-aware admission; preserve full-context freedom per request. Tune aggregate context pool + server parallelism P; durable retries/backpressure Q. Engine settled on **vLLM ≥0.27** (Blackwell/NVFP4); llama.cpp kept for router-aware offload of too-big models.
