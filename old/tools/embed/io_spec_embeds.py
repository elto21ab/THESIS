# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
#
# Raw IO spec for llama-server embedding/reranking modes.
# Shows exactly what goes in (request body) and what comes out (response shape).
#
#  MODE A  --embedding     POST /embedding    text(s) → vector(s)
#  MODE B  --reranking     POST /reranking    (query, docs) → relevance scalars
#                            1 doc  = pairwise cross-encode score
#                            N docs = ranked list
#
#  NOTE: there is no "cross-encode two docs without a query."
#    Cross-encoders always take (query, doc) — if you want a symmetric score
#    between two texts, use a biencoder and compute cosine between their vectors.
#
#  NOTE: --embedding --pooling rank is a low-level alternative to --reranking.
#    Instead of a /reranking endpoint you POST to /embedding with a hand-built
#    "query: ... document: ..." prompt and get back a 1-dim vector holding the score.
#    --reranking abstracts all of that away.
#
# Usage: `uv run io_spec_embeds.py`

import json, math, subprocess, time
from shutil import which

import requests

PORT = 8091          # different from testing_embeds.py (8090) so they can coexist
BASE = f"http://127.0.0.1:{PORT}"
M    = "/Users/e/.models/embed"

SEP = "─" * 58

def start(model, *flags):
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(0.5)
    p = subprocess.Popen(
        [which("llama-server"), "-m", model, "--port", str(PORT), *flags],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/health", timeout=2).ok: return p
        except: pass
        time.sleep(1)
    p.kill(); raise TimeoutError("server not ready")

def stop(p): p.terminate(); p.wait(timeout=5)
def vec(item): e = item["embedding"]; return e[0] if isinstance(e[0], list) else e
def cosine(a, b):
    d = sum(x*y for x,y in zip(a,b))
    return d / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))


# ── MODE A: text(s) → vector(s) ─────────────────────────────────────
p = start(f"{M}/bge-m3-q8_0.gguf", "--embedding")

req = {"content": ["The cat sat on the mat.", "Dogs are loyal companions."]}
res = requests.post(f"{BASE}/embedding", json=req).json()
v0, v1 = vec(res[0]), vec(res[1])

print(f"\n{SEP}")
print("MODE A  --embedding   POST /embedding")
print(f"{SEP}")
print(f"IN:   {json.dumps(req)}")
print(f"OUT:  list of {len(res)} items, each:")
print(f"        {{\"index\": int, \"embedding\": [[< {len(v0)} floats >]]}}")
print(f"      res[0]['embedding'][0][:4] = {[round(x,5) for x in v0[:4]]}")
print(f"      res[1]['embedding'][0][:4] = {[round(x,5) for x in v1[:4]]}")
print(f"NOTE: embedding is [[...]] (nested list) — the outer list is a ColBERT")
print(f"      artifact; for single-vector models just take embedding[0].")
print(f"      |v0|={math.sqrt(sum(x*x for x in v0)):.4f}  |v1|={math.sqrt(sum(x*x for x in v1)):.4f}  (L2-norm ≈ 1.0 = normalised)")
print(f"      cosine(v0, v1) = {cosine(v0,v1):+.4f}  (similarity you'd use for retrieval)")
stop(p)


# ── MODE B: (query, 1 doc) → pairwise scalar ────────────────────────
p = start(f"{M}/qwen3-reranker-0.6b-q8_0.gguf", "--reranking")

req = {"query": "small furry pet", "documents": ["A kitten on a rug."]}
res = requests.post(f"{BASE}/reranking", json=req).json()

print(f"\n{SEP}")
print("MODE B  --reranking   POST /reranking  (1 doc → pairwise score)")
print(f"{SEP}")
print(f"IN:   {json.dumps(req)}")
print(f"OUT:  {json.dumps(res)}")
print(f"NOTE: relevance_score is a raw logit (sigmoid output here ≈ 0–1).")
print(f"      NOT cosine. NOT comparable across different reranker models.")
stop(p)


# ── MODE C: (query, docs) → ranked list ─────────────────────────────
p = start(f"{M}/qwen3-reranker-0.6b-q8_0.gguf", "--reranking")

docs = ["The stock market crashed in 2008.",
        "A kitten was sitting on a rug.",
        "Quantum mechanics describes nature at the atomic scale."]
req  = {"query": "small furry pet", "documents": docs}
res  = requests.post(f"{BASE}/reranking", json=req).json()
ranked = sorted(res["results"], key=lambda x: x["relevance_score"], reverse=True)

print(f"\n{SEP}")
print("MODE C  --reranking   POST /reranking  (N docs → ranked list)")
print(f"{SEP}")
print(f"IN:   query = \"{req['query']}\"")
for i, d in enumerate(docs): print(f"        documents[{i}] = \"{d}\"")
print(f"OUT:  results (sorted by you, server returns original order):")
for r in ranked:
    print(f"        rank#{ranked.index(r)+1}  score={r['relevance_score']:.6f}  [{r['index']}] \"{docs[r['index']]}\"")
print(f"NOTE: same endpoint as MODE B — difference is just number of docs.")
stop(p)

print(f"\n{SEP}")
print("done.")
