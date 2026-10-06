"""Verify the OpenAI-compatible client against a local llama-server (Metal).

Bounded: starts the server, embeds a few texts, stops it. Uses bge-m3 (already on disk) so the
test needs no download; the Jina v5 retrieval GGUF is the same code path.
"""
import subprocess
import time

import requests

from retrieval.embed.openai import OpenAIEmbedder

MODEL = "/Users/e/.models/embed/bge-m3-q8_0.gguf"
PORT = 8093
BASE = f"http://127.0.0.1:{PORT}/v1"

srv = subprocess.Popen(
    ["llama-server", "-m", MODEL, "--embedding", "--port", str(PORT),
     "--host", "127.0.0.1", "-c", "2048", "-ub", "2048"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            if requests.get(f"http://127.0.0.1:{PORT}/health", timeout=2).ok:
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    else:
        raise SystemExit("server never became healthy")

    e = OpenAIEmbedder("bge-m3", base_url=BASE, dim=1024, task_prompt=False)
    docs = e.embed(["Hej med dig, hvordan går det?", "The cat sat on the mat."])
    for q in ["How are you doing?", "Hvor sidder katten?"]:
        v = e.embed([q], is_query=True)[0]
        print(f"{q!r:24} doc0(DA)={float(v @ docs[0]):+.3f}  doc1(EN)={float(v @ docs[1]):+.3f}")
    print("shape", docs.shape)
finally:
    srv.terminate()
    srv.wait(timeout=5)
