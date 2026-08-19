# /// script
# requires-python = ">=3.12"
# dependencies = ["requests"]
# ///
#
# Sanity-check embedding & reranking GGUF models via llama-server.
#
# Score guide:
#   Biencoder (--embedding): query and each doc are encoded independently into
#     vectors. We report COSINE SIMILARITY ∈ [-1, 1] between the query vector
#     and each doc vector.  When the model outputs L2-normalised vectors (|v|≈1),
#     cosine == dot-product — you'll often see this in the pooling=cls/mean case.
#   Cross-encoder (--reranking): query and doc are fed *together* and the model
#     outputs a single RELEVANCE SCALAR directly.  Scores are NOT cosine, not
#     bounded, and not comparable across different reranker models.
#
# Usage: `uv run testing_embeds.py`

import math, os, struct, subprocess, time
from shutil import which

import requests

DOCS = [
    "The cat sat on the mat.",
    "A kitten was sitting on a rug.",
    "Quantum mechanics describes nature at the atomic scale.",
    "The stock market crashed in 2008.",
    "Dogs are loyal companions.",
]
QUERY = "What is a small furry pet?"
PORT  = 8090
BASE  = f"http://127.0.0.1:{PORT}"

# ── GGUF metadata reader (no extra deps) ────────────────────────────
_POOL = {0: "none", 1: "mean", 2: "cls", 3: "last", 4: "rank"}

def gguf_meta(path: str) -> dict:
    _fmts = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
    def rd(f, fmt): return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    def rstr(f):    return f.read(rd(f, '<Q')).decode(errors='replace')
    def rval(f, t):
        if t == 8: return rstr(f)
        if t == 9: et = rd(f,'<I'); return [rval(f,et) for _ in range(rd(f,'<Q'))]
        return rd(f, _fmts[t])
    with open(path, 'rb') as f:
        f.read(4); rd(f,'<I'); rd(f,'<Q'); n_kv = rd(f,'<Q')
        kv = {rstr(f): rval(f, rd(f,'<I')) for _ in range(n_kv)}
    arch = kv.get("general.architecture", "")
    return {
        "arch":    arch,
        "ctx":     kv.get(f"{arch}.context_length", kv.get("llama.context_length", "?")),
        "pooling": _POOL.get(kv.get(f"{arch}.pooling_type", kv.get("llama.pooling_type")), "default"),
    }

# ── server ───────────────────────────────────────────────────────────
def serve(model: str, flags: list[str]):
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    proc = subprocess.Popen(
        [which("llama-server"), "-m", model, "--port", str(PORT)] + flags,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return proc
        except requests.ConnectionError: pass
        if proc.poll() is not None: raise RuntimeError(f"server died (exit {proc.returncode})")
        time.sleep(1)
    proc.kill(); raise TimeoutError("server not ready after 60s")

def kill(proc):
    proc.terminate()
    try: proc.wait(timeout=5)
    except subprocess.TimeoutExpired: proc.kill()

# ── helpers ──────────────────────────────────────────────────────────
def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(x*x for x in b)))

def _vec(item):
    e = item["embedding"]
    return e[0] if isinstance(e[0], list) else e

def _header(kind, model, flags, meta, dim, q_norm=None):
    pool_flag = next((flags[i+1] for i,f in enumerate(flags) if f == "--pooling" and i+1 < len(flags)), None)
    pooling   = meta["pooling"] + (f" → {pool_flag} (override)" if pool_flag else "")
    norm_str  = f"  ·  |q|={q_norm:.3f}" if q_norm is not None else ""
    print(f"\n{'═'*62}")
    print(f"  {kind}")
    print(f"  model:   {os.path.basename(model)}")
    print(f"  flags:   {' '.join(flags)}")
    print(f"  arch: {meta['arch']}  ·  ctx: {meta['ctx']}  ·  pooling: {pooling}  ·  dim: {dim}{norm_str}")
    #print(f"{'═'*62}")

# ── tests ────────────────────────────────────────────────────────────
def test_embed(model, flags=None, label="Biencoder · vector encoder"):
    flags = list(flags or []) + ["--embedding"]
    meta  = gguf_meta(model)
    proc  = serve(model, flags)
    try:
        vecs   = requests.post(f"{BASE}/embedding", json={"content": [QUERY] + DOCS}, timeout=120).json()
        q_vec  = _vec(vecs[0])
        q_norm = math.sqrt(sum(x*x for x in q_vec))
        _header(label, model, flags, meta, len(q_vec), q_norm)
        print(f"  query: \"{QUERY}\"")
        ranked = sorted(enumerate(_vec(v) for v in vecs[1:]), key=lambda x: cosine(q_vec, x[1]), reverse=True)
        for rank, (i, v) in enumerate(ranked, 1):
            print(f"  #{rank}  cos={cosine(q_vec, v):+.4f}  \"{DOCS[i]}\"")
    finally:
        kill(proc)

def test_rerank(model, flags=None, label="Cross-encoder · reranker"):
    flags = list(flags or []) + ["--reranking"]
    meta  = gguf_meta(model)
    proc  = serve(model, flags)
    try:
        r      = requests.post(f"{BASE}/reranking", json={"query": QUERY, "documents": DOCS}, timeout=120).json()
        ranked = sorted(r.get("results", []), key=lambda x: x.get("relevance_score", 0), reverse=True)
        _header(label, model, flags, meta, "—")
        print(f"  query: \"{QUERY}\"")
        for rank, item in enumerate(ranked, 1):
            print(f"  #{rank}  score={item['relevance_score']:.6f}  \"{DOCS[item['index']]}\"")
    finally:
        kill(proc)


if __name__ == "__main__":
    M = os.path.expanduser("~/.models/embed")

    # ── TOGGLE: uncomment/comment tests ──────────────────────────────
    test_embed(f"{M}/bge-m3-q8_0.gguf",              label="Biencoder · vector encoder  [bge-m3]")
    test_embed(f"{M}/embeddinggemma-300M-Q8_0.gguf",  label="Biencoder · vector encoder  [gemma-300M]")
    # test_embed(f"{M}/bge-m3-q8_0.gguf", ["--pooling", "mean"],  label="Biencoder [bge-m3, mean pool]")  # ← only if model was trained w/ mean
    # test_embed(f"{M}/bge-m3-q8_0.gguf", ["--pooling", "cls"],   label="Biencoder [bge-m3, cls pool]")
    # test_embed(f"{M}/bge-m3-q8_0.gguf", ["--pooling", "last"],  label="Biencoder [bge-m3, last pool]")

    test_rerank(f"{M}/qwen3-reranker-0.6b-q8_0.gguf", label="Cross-encoder · reranker  [qwen3-0.6b]")
    # test_rerank(f"{M}/bge-m3-q8_0.gguf", ["--pooling", "rank"],  label="Biencoder used as reranker  [bge-m3]")

    print("\ndone.")
