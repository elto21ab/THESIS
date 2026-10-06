"""vllm-metal half of the Qwen3 3-way. Run with ~/.venv-vllm-metal/bin/python.

Reads {"query", "docs"} on stdin, prints one timing line. First run downloads the MLX checkpoint.
All work is under __main__: vLLM spawns a child that re-imports this module, and a top-level
stdin read would blow up there (macOS uses 'spawn', not 'fork').
"""
import json
import sys
import time

REPS = 3


def main() -> None:
    from vllm import LLM

    payload = json.load(sys.stdin)
    q, ds = payload["query"], payload["docs"]

    llm = LLM(
        model="mku64/Qwen3-Reranker-0.6B-mlx-8Bit",
        revision="ba80418a47fa1c4368a6c2287b0e449904063576",
        runner="pooling",
        max_model_len=512,
        gpu_memory_utilization=0.35,
        enforce_eager=True,
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
    )

    t = time.time()
    llm.score(q, ds)
    cold = time.time() - t
    t = time.time()
    for _ in range(REPS):
        llm.score(q, ds)
    warm = (time.time() - t) / REPS
    print(f"{'vllm-metal':16} cold={cold:6.2f}s  per-call={warm:5.2f}s  {warm/len(ds)*1000:6.0f} ms/doc")


if __name__ == "__main__":
    main()
