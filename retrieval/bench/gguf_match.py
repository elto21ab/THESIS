"""Step-1 acceptance: does the Jina v5 retrieval GGUF (llama.cpp) reproduce the PyTorch model?

Assumes `llama-server --embedding` on 8091 serves the Q8_0 GGUF. Compares cosines between the two
backends on identical texts, plus cross-lingual ranking. Backends must agree or dev/prod differ.
"""
import numpy as np

from retrieval.embed.local import JinaLocalEmbedder
from retrieval.embed.openai import OpenAIEmbedder

TEXTS = [
    "Hej med dig, hvordan går det?",
    "The cat sat on the mat.",
    "Jeg har glemt mine handsker hos dig.",
    "Are we still meeting at 21?",
]

g = OpenAIEmbedder("jina-v5-retrieval", base_url="http://127.0.0.1:8091/v1", task_prompt=True)
t = JinaLocalEmbedder(dim=1024, max_length=2048)
vg, vt = g.embed(TEXTS), t.embed(TEXTS)
for i, s in enumerate(TEXTS):
    print(f"gguf~torch doc{i} = {float(vg[i] @ vt[i]):+.3f}   {s[:40]!r}")

for q in ["How are you doing?", "Hvor sidder katten?"]:
    qg, qt = g.embed([q], is_query=True)[0], t.embed([q], is_query=True)[0]
    rg = np.argsort(-(vg @ qg))[:2]
    rt = np.argsort(-(vt @ qt))[:2]
    print(f"{q!r:24} gguf top2={list(rg)}  torch top2={list(rt)}  agree={list(rg) == list(rt)}")
