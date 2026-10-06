"""OpenAI-compatible embeddings client.

Same client drives both:
  - local llama.cpp `llama-server --embedding` (Metal, dev)  -> base_url http://127.0.0.1:8091/v1
  - remote vLLM `--task embed` (prod, UCloud)                -> base_url http://host:port/v1

Jina v5 is a task model: inputs are prefixed `Query: ` / `Document: `, matching the PyTorch
encode(prompt_name=...). The `-retrieval` GGUF bakes the task adapter, so this prefix is the only
per-call switch.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import normalize


class OpenAIEmbedder:
    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:8091/v1",
                 api_key: str = "sk-local", dim: int = 1024, batch: int = 32,
                 task_prompt: bool = True, timeout: float = 300.0):
        import requests  # lazy: only needed when this backend is used
        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.dim = dim
        self.batch = batch
        self.task_prompt = task_prompt
        self.timeout = timeout

    def count(self, text: str) -> int:
        # No local tokenizer over HTTP; a char proxy is fine for dev (chunk budgets, not billing).
        return max(1, len(text) // 4)

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        prefix = ""
        if self.task_prompt:
            prefix = "Query: " if is_query else "Document: "
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
        return normalize(v)
