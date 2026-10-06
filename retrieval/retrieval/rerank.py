"""Cross-encoder reranker over HTTP.

One client, two servers:
  - llama.cpp `llama-server --reranking`  (Metal, dev)  -> path /reranking
  - vLLM / Jina rerank API (prod)                        -> path /v1/rerank
Both return {"results":[{"index","relevance_score"}...]}.
"""
from __future__ import annotations

from typing import Protocol, Sequence


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]: ...


class HttpReranker:
    def __init__(self, *, base_url: str = "http://127.0.0.1:8092",
                 model: str = "jina-reranker-v3.5", path: str = "/reranking",
                 api_key: str = "sk-local", timeout: float = 300.0):
        import requests
        self._requests = requests
        self.url = base_url.rstrip("/") + path
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def rerank(self, query: str, docs: Sequence[str], *,
               top_k: int | None = None) -> list[tuple[int, float]]:
        r = self._requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "query": query, "documents": list(docs)},
            timeout=self.timeout,
        )
        r.raise_for_status()
        res = sorted(r.json()["results"], key=lambda d: -d["relevance_score"])
        out = [(int(d["index"]), float(d["relevance_score"])) for d in res]
        return out[:top_k] if top_k else out
