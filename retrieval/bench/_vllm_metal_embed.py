"""vllm-metal embeddings for jina-embeddings-v5-text-small-retrieval. Own venv.

Usage: _vllm_metal_embed.py texts.json   (JSON list of strings)
Prints: n, dim, total seconds, ms/item.
All work under __main__ (vLLM spawns a child that re-imports this module).
"""
import json
import os
import sys
import time

MODEL = os.environ.get("VLLM_EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small-retrieval")


def main() -> None:
    from vllm import LLM

    texts = json.load(open(sys.argv[1]))
    llm = LLM(model=MODEL, runner="pooling", max_model_len=2048, convert="embed",
              gpu_memory_utilization=0.35, enforce_eager=True,
              # Jina's ST config omits rope_theta; the base model uses 3.5e6.
              hf_overrides={"rope_theta": 3_500_000.0})
    mc = llm.llm_engine.model_config
    print(f"convert_type={mc.convert_type!r}  supported_tasks={mc.supported_tasks}", flush=True)
    llm.embed(texts[:1])  # warmup / load
    t = time.time()
    out = llm.embed(texts)
    dt = time.time() - t
    dim = len(out[0].outputs.embedding)
    print(f"{'vllm-metal embed':16} n={len(out)} dim={dim} total={dt:.2f}s  {dt/len(out)*1000:.1f} ms/item")


if __name__ == "__main__":
    main()
