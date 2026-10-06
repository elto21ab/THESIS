"""Local Jina v5 embedder (transformers + MPS). Heavy imports are lazy.

API per model card: model.encode(texts, task=..., prompt_name="query"|"passage",
truncate_dim=..., max_length=...). Prod swaps this for the vLLM OpenAI-compatible client.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

MODEL = "jinaai/jina-embeddings-v5-text-small"


class JinaLocalEmbedder:
    def __init__(self, name: str = MODEL, *, task: str = "retrieval", dim: int = 1024,
                 max_length: int = 8192, device: str | None = None, batch: int = 16,
                 dtype: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        kw = {"dtype": getattr(torch, dtype)} if dtype else {}
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True, **kw)
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(device).eval()
        self.dim = dim
        self.task = task
        self.max_length = max_length
        self.batch = batch

    def count(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])

    def embed(self, texts: Sequence[str], *, is_query: bool = False) -> np.ndarray:
        prompt = "query" if is_query else "document"
        vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.batch):
            chunk = list(texts[i:i + self.batch])
            out = self.model.encode(texts=chunk, task=self.task, prompt_name=prompt,
                                    truncate_dim=self.dim, max_length=self.max_length)
            if hasattr(out, "detach"):  # torch tensor on MPS -> float32 on host before numpy
                out = out.detach().float().cpu()
            vecs.append(np.asarray(out, dtype=np.float32))
        return np.vstack(vecs) if vecs else np.zeros((0, self.dim), dtype=np.float32)
