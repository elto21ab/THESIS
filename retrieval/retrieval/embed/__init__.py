from .base import Embedder, StubEmbedder, normalize


def get_embedder(name: str, **kw) -> Embedder:
    if name == "stub":
        return StubEmbedder()
    if name == "jina":
        from .local import JinaLocalEmbedder
        return JinaLocalEmbedder(**kw)
    if name == "openai":  # llama-server (local) or vLLM (remote)
        from .openai import OpenAIEmbedder
        return OpenAIEmbedder(**kw)
    raise ValueError(f"unknown embedder: {name!r} (stub | jina | openai)")
