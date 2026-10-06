"""Qwen3-Embedding-0.6B through vllm-metal (MLX 8-bit). Own venv.

Usage: _vllm_metal_qwen_embed.py texts.json
"""
import json
import sys
import time

MODEL = "mlx-community/Qwen3-Embedding-0.6B-8bit"


def main() -> None:
    from vllm import LLM

    texts = json.load(open(sys.argv[1]))
    llm = LLM(model=MODEL, runner="pooling", max_model_len=2048,
              gpu_memory_utilization=0.35, enforce_eager=True)
    llm.embed(texts[:1])
    t = time.time()
    out = llm.embed(texts)
    dt = time.time() - t
    print(f"{'vllm-metal Qwen3-emb':22} n={len(out)} dim={len(out[0].outputs.embedding)} "
          f"total={dt:.2f}s  {dt/len(out)*1000:.1f} ms/item")


if __name__ == "__main__":
    main()
