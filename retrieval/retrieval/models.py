"""HTTP clients for the two model stages: embeddings and reranking.

Both speak the OpenAI-compatible API, so the same code serves local (vllm-metal, llama.cpp) and
prod (vLLM on CUDA) — only the served model differs (see config.<profile>.yaml).

Prompt styles (`prompt:`) are a property of the *model*, not the server: the server embeds/scores
whatever text it is handed, so the instruction wrapper is assembled here.
  jina  — Jina v5:      query `Query: `,            document `Document: `
  qwen3 — Qwen3:        query `Instruct: {task}\\nQuery: {q}`, document plain
  none  — bge-m3 and the Jina reranker API: no wrapper
Rerankers take the pair `<Instruct>: … <Query>: … <Document>: …` (qwen3) or nothing (jina).
"""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

TASK = "Given a question about the author's own life, retrieve chat messages that answer it"


def _normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


class Embedder(Protocol):
    dim: int

    def count(self, text: str) -> int: ...
    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray: ...


def query_prefix(style: str, *, task: str = TASK) -> str:
    return {"jina": "Query: ", "qwen3": f"Instruct: {task}\nQuery: "}.get(style, "")


def doc_prefix(style: str) -> str:
    return {"jina": "Document: "}.get(style, "")


def rerank_pair(style: str, query: str, doc: str, *, task: str = TASK) -> tuple[str, str]:
    if style == "qwen3":
        return f"<Instruct>: {task}\n<Query>: {query}\n<Document>: ", doc
    return query, doc


class OpenAIEmbedder:
    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:8091/v1",
                 api_key: str = "sk-local", dim: int = 1024, batch: int = 32,
                 prompt: str = "none", max_length: int | None = None, timeout: float = 300.0):
        import requests  # lazy: only needed when this backend is used
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.dim = dim
        self.batch = batch
        self.prompt = prompt
        self.max_length = max_length
        self.timeout = timeout

    def count(self, text: str) -> int:
        # No local tokenizer over HTTP; a char proxy is fine for chunk budgets, not billing.
        return max(1, len(text) // 4)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        prefix = query_prefix(self.prompt) if is_query else doc_prefix(self.prompt)
        out: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            chunk = [prefix + t for t in texts[i:i + self.batch]]
            r = self._requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": chunk},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            out.append(np.asarray([d["embedding"] for d in data], dtype=np.float32))
        v = np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(v)


class PlaceholderEmbedder:
    """Words hashed into a 256-d bag-of-words. No model, no semantics, no download.

    Its only job is to exercise the *vector* code path offline (`embed.backend: placeholder`). It
    cannot measure retrieval quality — use a real embedder, or a lexical index for offline work."""

    dim = 256

    def __init__(self, prompt: str = "none") -> None:
        self.prompt = prompt

    def count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        import hashlib
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            for word in t.lower().split():
                h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "big")
                out[r, h % self.dim] += 1.0
        return _normalize(out)


def get_embedder(backend: str, **kw) -> Embedder | None:
    """None means 'no embedder': a lexical index needs no model."""
    if backend in (None, "lexical", "none"):
        return None
    if backend == "placeholder":
        return PlaceholderEmbedder(prompt=kw.get("prompt", "none"))
    if backend == "openai":
        return OpenAIEmbedder(kw.pop("model"), **kw)
    raise ValueError(f"unknown embed backend: {backend!r} (openai | placeholder | lexical)")


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]: ...


class HttpReranker:
    """Rerank over HTTP. llama.cpp `llama-server --reranking` → `/reranking`;
    vLLM / Jina API → `/v1/rerank`. Both return {"results":[{"index","relevance_score"}…]}."""

    def __init__(self, *, model: str, base_url: str = "http://127.0.0.1:8092",
                 path: str = "/reranking", prompt: str = "none", api_key: str = "sk-local",
                 timeout: float = 300.0):
        import requests
        self._requests = requests
        self.url = base_url.rstrip("/") + path
        self.model = model
        self.prompt = prompt
        self.api_key = api_key
        self.timeout = timeout

    def rerank(self, query: str, docs: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]:
        q, _ = rerank_pair(self.prompt, query, "")
        payload_docs = [rerank_pair(self.prompt, query, d)[0] if self.prompt == "qwen3" else d
                        for d in docs]
        r = self._requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "query": q, "documents": payload_docs},
            timeout=self.timeout,
        )
        r.raise_for_status()
        res = sorted(r.json()["results"], key=lambda d: -d["relevance_score"])
        out = [(int(d["index"]), float(d["relevance_score"])) for d in res]
        return out[:top_k] if top_k else out
