"""The agent loop: model + tools + state. Caller-side scaffolding, deliberately small.

    push   pack() once before the first model call — the default context injection
    pull   `search` / `get` exposed as tools; the `get.ref` enum is rebuilt after every search
    state  RefRegistry (ref → hit), a message-level seen-set (dedupe across turns), a running
           injected-token budget, a tool-call cap and a turn cap

Needs an OpenAI-compatible chat endpoint (vLLM / llama.cpp). Everything model- or index-specific
comes from config (`llm:` and `embed:`), so local and prod differ by file, not by code.

Two invariants worth keeping: a message is never shown twice (a repeat reads as emphasis the donor
never gave), and the model only ever names a ref it has actually been handed — which is why the
`get` tool schema is regenerated with an `enum` after each search.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .index import Index
from .models import Embedder, Reranker
from .ops import (MsgKey, RefRegistry, get, get_tool_schema, hit_payload, mark_shown, pack,
                  render_hits, search_tool_schema)
from .search import search_any

SYSTEM = """You answer a survey as {donor}, grounding every claim in messages retrieved from
{donor}'s own chat history. Use the `search` tool to find evidence and `get` to read more of a
message region you already found. If the history says nothing relevant, say so plainly instead of
inventing. Answer in the language of the question, first person, short — a few sentences."""


@dataclass
class LoopResult:
    answer: str
    turns: int
    calls: list[dict] = field(default_factory=list)   # [{tool, args, hits}] for the trace
    injected_tokens: int = 0
    stopped: str = "answered"                         # | max_turns | max_calls | budget


def _post(base_url: str, api_key: str, payload: dict, *, timeout: float = 600.0) -> dict:
    import requests
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run(index: Index, question: str, *, cfg: dict, embedder: Embedder | None = None,
        reranker: Reranker | None = None, push: bool = False, pull: bool = True,
        k: int | None = None, dry_run: bool = False) -> LoopResult:
    llm = cfg["llm"]
    ret = cfg["retrieve"]
    k = k or ret["k"]
    donor = index.manifest.get("donor_id") or index.manifest.get("donor") or "the donor"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM.format(donor=donor)},
    ]
    seen: set[MsgKey] = set()
    reg = RefRegistry()
    injected = 0
    calls: list[dict] = []

    if push:  # prefill: one pack covering the survey probes
        p = pack([question], lambda q, n: search_any(
            index, q, embedder, k=n, reranker=reranker,
            max_region_chunks=ret["max_region_chunks"]),
            k=ret["k"], budget_tokens=ret["budget"], min_threads=ret["min_threads"])
        messages.append({"role": "user",
                         "content": "Context retrieved from your own chat history:\n\n"
                                    + "\n\n".join(p.texts)})
        injected += p.stats["tokens"]
        for h in p.hits:
            reg.add([h])
    messages.append({"role": "user", "content": question})

    if dry_run:
        return LoopResult(answer="", turns=0, injected_tokens=injected,
                          calls=[{"dry_run": {"messages": messages,
                                              "tools": _tools(reg, pull)}}])

    for turn in range(int(llm["max_turns"])):
        payload = {"model": llm["model"], "messages": messages, "temperature": llm["temperature"],
                   "tools": _tools(reg, pull), "tool_choice": "auto" if pull else "none"}
        resp = _post(llm["base_url"], llm.get("api_key", "sk-local"), payload)
        msg = resp["choices"][0]["message"]
        calls_new = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "",
                         **({"tool_calls": calls_new} if calls_new else {})})
        if not calls_new:
            return LoopResult(msg.get("content") or "", turn + 1, calls, injected, "answered")
        if len(calls) + len(calls_new) > int(llm["max_calls"]):
            return LoopResult(msg.get("content") or "", turn + 1, calls, injected, "max_calls")

        for call in calls_new:
            name = call["function"]["name"]
            args = json.loads(call["function"].get("arguments") or "{}")
            result, used = _dispatch(index, name, args, reg, embedder, reranker, ret, seen, k)
            injected += used
            calls.append({"tool": name, "args": args, "result_bytes": len(json.dumps(result))})
            messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                             "content": json.dumps(result, ensure_ascii=False)})
        if injected > ret["budget"]:
            return LoopResult("", turn + 1, calls, injected, "budget")
    return LoopResult("", int(llm["max_turns"]), calls, injected, "max_turns")


def _tools(reg: RefRegistry, pull: bool) -> list[dict]:
    """`get.ref` is constrained to the refs handed out so far — regenerated every turn, so the
    model physically cannot name a region it has not seen."""
    if not pull:
        return []
    return [search_tool_schema(), reg.get_tool_schema()] if len(reg) else [search_tool_schema()]


def _dispatch(index: Index, name: str, args: dict, reg: RefRegistry, embedder, reranker,
              ret: dict, seen: set[MsgKey], k: int) -> tuple[dict, int]:
    if name == "search":
        hits = search_any(index, args.get("query", ""), embedder,
                          k=int(args.get("k") or k), reranker=reranker,
                          max_region_chunks=ret["max_region_chunks"])
        texts, _ = render_hits(hits, seen=seen, mark=False)
        reg.add(hits)
        outs = [hit_payload(h, max_chars=ret["max_chars"], rank=i, text=t)
                for i, (h, t) in enumerate(zip(hits, texts), 1)]
        mark_shown(zip(hits, [o["text"] for o in outs]), seen)   # only what the model actually read
        payload = {"donor": index.manifest.get("donor_id"), "hits": outs}
        return payload, sum(len(o["text"]) for o in outs) // 4
    if name == "get":
        hit = reg.resolve(args.get("ref", 0))
        if hit is None:
            return {"error": "unknown ref"}, 0
        window = int(args.get("window") or 0)
        if window:  # raw messages around the region
            msgs = get(index, hit.chunk.id, window=window, bundle=index.manifest.get("bundle"))
            lines = msgs if isinstance(msgs, list) else []
            fresh = [m for m in lines
                     if (hit.platform, hit.partner, m["i"]) not in seen]  # skip shown messages
            for m in fresh:
                seen.add((hit.platform, hit.partner, m["i"]))
            return {"id": hit.chunk.id, "window": window, "msgs": fresh}, sum(
                len(m["text"]) for m in fresh) // 4
        texts, _ = render_hits([hit], seen=seen, mark=False)
        out = hit_payload(hit, max_chars=0, rank=args.get("ref"), text=texts[0])
        mark_shown([(hit, out["text"])], seen)
        return out, len(out["text"]) // 4
    return {"error": f"unknown tool {name!r}"}, 0
