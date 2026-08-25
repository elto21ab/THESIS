# UCloud inference optimization

Start here:

1. [`docs/DECISIONS_AND_NEXT_SESSION.md`](docs/DECISIONS_AND_NEXT_SESSION.md) — current decisions, corrections, metrics, next implementation/run.
2. [`docs/INFERENCE_TUNING_HANDOFF.md`](docs/INFERENCE_TUNING_HANDOFF.md) — full hardware/history/reproduction handoff.
3. [`checkpoints/run-12372444/RESULTS_8B.md`](checkpoints/run-12372444/RESULTS_8B.md) — compact validated 8B results.

Folders:

- `tools/`: tuner + prompt-preparation CLI
- `configs/`: example TOML
- `scripts/`: UCloud bootstrap
- `data/`: generated benchmark prompt suites; original V7 source remains elsewhere
- `checkpoints/`: compact local experiment artifacts, incl. historical Qwen run
- `docs/`: handoff and decisions

Policy: no token-aware admission. Preserve full-context freedom per request; tune aggregate context pool + server parallelism P; use durable retries/backpressure Q.
