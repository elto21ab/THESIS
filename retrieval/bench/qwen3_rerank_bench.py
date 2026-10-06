"""Qwen3-Reranker-0.6B, three runtimes, same 24 docs.

  transformers : Qwen/Qwen3-Reranker-0.6B, bf16, yes/no logit score, one forward per doc
  llama.cpp    : llama-server --reranking, Q8_0, one listwise call
  vllm-metal   : mku64/Qwen3-Reranker-0.6B-mlx-8Bit (8-bit), LLM.score, in its own venv

Weights differ per runtime (bf16 vs Q8_0 vs 8-bit MLX) — this measures runtime, not parity.
"""
import json
import subprocess
import time
from pathlib import Path

import requests

INDEX = Path("out/jina-gguf-sample200/chunks.jsonl")
N_DOCS, REPS = 24, 3
VLLM_PY = Path.home() / ".venv-vllm-metal/bin/python"
HERE = Path(__file__).parent


def docs():
    cs = [json.loads(l) for l in INDEX.read_text().splitlines()]
    out = []
    for c in cs:
        body = " ".join(l for l in c["text"].split("\n") if l.startswith("["))
        w = body.split()
        if len(w) >= 25:
            out.append(" ".join(w[:70]))
        if len(out) == N_DOCS:
            break
    return out


def timeit(name, fn, q, ds):
    t = time.time()
    fn(q, ds)  # warmup / load
    cold = time.time() - t
    t = time.time()
    for _ in range(REPS):
        fn(q, ds)
    warm = (time.time() - t) / REPS
    print(f"{name:16} cold={cold:6.2f}s  per-call={warm:5.2f}s  {warm/len(ds)*1000:6.0f} ms/doc", flush=True)


def main():
    q, ds = "Hvad lavede vi i weekenden?", docs()

    def llamacpp(q, d):
        r = requests.post("http://127.0.0.1:8092/reranking", timeout=600,
                          json={"model": "qwen3-reranker-0.6b", "query": q, "documents": d})
        r.raise_for_status()
        return r.json()["results"]

    try:
        timeit("llama.cpp", llamacpp, q, ds)
    except Exception as exc:  # noqa: BLE001
        print(f"llama.cpp FAILED: {exc}")

    try:  # vllm-metal in its own venv via a helper
        p = subprocess.run([str(VLLM_PY), str(HERE / "_qwen3_vllm_metal.py")],
                           input=json.dumps({"query": q, "docs": ds}),
                           capture_output=True, text=True, timeout=1800)
        print(p.stdout.strip() or f"vllm-metal FAILED: {p.stderr.strip()[-300:]}")
    except Exception as exc:  # noqa: BLE001
        print(f"vllm-metal FAILED: {exc}")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen3-Reranker-0.6B"
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = model.to(dev).eval()
        yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
        head = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
                'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
                '<|im_end|>\n<|im_start|>user\n')
        tail = '<|im_end|>\n<|im_start|>assistant\n'
        instr = "Given a web search query, retrieve relevant passages that answer the query"

        def hf(q, d):
            for doc in d:
                text = f"{head}<Instruct>: {instr}\n<Query>: {q}\n<Document>: {doc}{tail}"
                ids = tok(text, return_tensors="pt").to(dev)
                with torch.no_grad():
                    logits = model(**ids).logits[:, -1, :]
                torch.softmax(torch.stack([logits[0, no], logits[0, yes]]), dim=0)[1].item()
            return None

        timeit("transformers/mps", hf, q, ds)
    except Exception as exc:  # noqa: BLE001
        print(f"transformers FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
