"""Jina v3.5 reranker, BF16, llama.cpp-fork vs transformers. Same docs, same warmup.

llama.cpp path : rerank.py (GGUFReranker) -> forked llama-embedding, one call per block.
transformers   : AutoModel.rerank, bf16 on MPS.

Two numbers: steady-state (model loaded, per rerank call) and cold (includes load).
"""
import json
import sys
import time
from pathlib import Path

GGUF_DIR = Path.home() / ".models/embed/jina-reranker-v3.5"
BIN = Path.home() / "builds/llama.cpp-jina/build/bin/llama-embedding"
HF_DIR = Path.home() / ".models/embed/jina-reranker-v3.5-hf"
INDEX = Path("out/jina-gguf-sample200/chunks.jsonl")

N_DOCS = 24
REPS = 3


def docs():
    if INDEX.exists():
        cs = [json.loads(l) for l in INDEX.read_text().splitlines()]
        texts = []
        for c in cs:
            body = " ".join(l for l in c["text"].split("\n") if l.startswith("["))
            w = body.split()
            if len(w) >= 25:
                texts.append(" ".join(w[:70]))
            if len(texts) == N_DOCS:
                break
        if len(texts) == N_DOCS:
            return texts
    return [f"Document {i}: " + ("the study of topic %d and its effects " % i) * 12 for i in range(N_DOCS)]


def bench(name, fn, docs, query):
    t = time.time()
    fn(query, docs)  # warmup / load
    cold = time.time() - t
    t = time.time()
    for _ in range(REPS):
        out = fn(query, docs)
    warm = (time.time() - t) / REPS
    print(f"{name:14} cold={cold:6.2f}s  per-call={warm:5.2f}s  {warm/len(docs)*1000:6.0f} ms/doc", flush=True)
    return out


def main():
    query = "Hvad lavede vi i weekenden?"
    ds = docs()

    try:
        sys.path.insert(0, str(GGUF_DIR))
        from rerank import GGUFReranker
        r = GGUFReranker(model_path=str(GGUF_DIR / "jina-reranker-v3.5-BF16.gguf"),
                         projector_path=str(GGUF_DIR / "projector.safetensors"),
                         llama_embedding_path=str(BIN),
                         tokenizer_path=str(GGUF_DIR / "tokenizer.json"))
        bench("llama.cpp", lambda q, d: r.rerank(q, d), ds, query)
    except Exception as exc:  # noqa: BLE001
        print(f"llama.cpp FAILED: {type(exc).__name__}: {exc}")

    try:
        import torch
        from transformers import AutoModel
        m = AutoModel.from_pretrained(str(HF_DIR), dtype="auto", trust_remote_code=True)
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        m = m.to(dev).eval()
        bench(f"transformers/{dev}", lambda q, d: m.rerank(q, d), ds, query)
    except Exception as exc:  # noqa: BLE001
        print(f"transformers FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
