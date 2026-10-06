"""Mainline vLLM 0.30.0 on CPU (metal plugin disabled) with jina-reranker-v3.5.

Tests whether JinaForRanking loads and scores on this Mac without the Metal plugin.
Run from the vllm-metal venv (it already holds vLLM 0.30.0+cpu):
    VLLM_PLUGINS= ~/.venv-vllm-metal/bin/python bench/_vllm_cpu_jina.py
"""
import time


def main() -> None:
    from vllm import LLM

    llm = LLM(
        model="jinaai/jina-reranker-v3.5",
        runner="pooling",
        max_model_len=2048,
        enforce_eager=True,
        gpu_memory_utilization=0.35,  # on the CPU backend this caps CPU RAM reserved
    )
    docs = ["A kitten on a rug.", "The stock market crashed.", "A cat sleeps."]
    out = llm.score("small furry pet", docs)
    for d, o in zip(docs, out):
        print(f"  {o.outputs.score:+.4f}  {d}", flush=True)

    t = time.time()
    llm.score("small furry pet", docs)
    print(f"one 3-doc call: {time.time()-t:.2f}s", flush=True)


if __name__ == "__main__":
    main()
