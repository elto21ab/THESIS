"""The agent loop: model + two verbs + state. One file, no tool schemas.

    search <query>        search the donor's chat history
    get <ref> [±N]        read around a region you were already given
    answer <text>         stop and answer

Why text and not function calling: the loop has to run the same on a 0.8B model as on a 27B one, and
only the big ones emit well-formed `tool_calls`. A 0.8B model that cannot call a tool can still write
`search zumba` on a line — and the parser below is the whole protocol. It is the mini-SWE-agent
bargain: the model's action space is a command line, the environment's answer is text, and the
harness owns everything that must not be wrong (which refs exist, what has been shown, the budget).

What the harness owns, in order of how much it matters:

* **refs** — the model never sees a chunk id. `R3` is registered when `search` returns it and lives
  until the run ends. A ref it was never handed cannot be resolved, so the failure mode is a message
  saying so, not a fabricated quote.
* **the seen-set** — a message is shown once per run. The 40% chunk overlap means neighbouring hits
  repeat a third of their lines; folding those out is most of the token saving, and it also stops a
  repeated line from reading as emphasis the donor never gave.
* **the budget** — estimated tokens of injected text, capped. The best hit is always kept, because a
  survey question that finds one relevant region should still get it.
* **caps** — turns and tool calls, so a model that loops on `search` ends instead of burning an hour.

Two invariants are load-bearing. A message is never shown twice. And the observation for a `search`
is rendered from the chunk text the index already holds — no second renderer, no re-reading the
bundle.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .index import Index
from .models import Embedder, Reranker
from .ops import (MsgKey, RefRegistry, get, hit_payload, mark_shown, push as push_block,
                  render_hits, window_lines)
from .search import search_any
from .tokens import DEFAULT_RATIO, count as _count

SYSTEM = """You answer a survey as {donor}, using only messages from {donor}'s own chat history.

You have three commands. Write exactly one per message, on its own line:

  search <words>      search the history. Returns numbered regions R1, R2, … with the messages.
  get <ref> [±N]      read region R3 again, or N messages around it: `get 3 ±5`
  answer <text>       stop searching and answer the question.

Rules that matter:
- Ground every claim in a message you actually read. If the history does not answer the question, say
  so with `answer` instead of inventing.
- Prefer one good region over many shallow ones. Search again only when the last result was beside
  the point.
- Write the answer in the language of the question, first person, a few sentences.

The search words are yours to choose from the question — two or three words the donor would have
typed, not the whole question.
"""

#: `search zumba`, `get 3 ±5`, `get 3`, `answer I usually …`. The last command in a message wins:
#: models like to narrate ("Let me look… search zumba") and the narration is not the action.
CMD = re.compile(r"^[ \t>*\-]*(search|get|answer)\b[: ]?[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
GET_ARG = re.compile(r"(\d+)\s*(?:[±+\-]\s*(\d+))?", re.UNICODE)


@dataclass
class LoopResult:
    answer: str
    turns: int
    calls: list[dict] = field(default_factory=list)   # [{tool, args, result_bytes}] for the trace
    injected_tokens: int = 0
    stopped: str = "answered"                         # | max_turns | max_calls | budget


@dataclass(slots=True)
class State:
    index: Index
    refs: RefRegistry
    seen: set[MsgKey]
    ret: dict
    k: int
    embedder: Embedder | None
    reranker: Reranker | None
    injected: int = 0
    calls: list[dict] = field(default_factory=list)
    ratio: float = DEFAULT_RATIO

    def search(self, query: str) -> str:
        hits = search_any(self.index, query, self.embedder, k=self.k, reranker=self.reranker,
                          max_region_chunks=self.ret["max_region_chunks"])
        if not hits:
            return "no messages matched. Try fewer words, or a word the donor would have used."
        texts, _ = render_hits(hits, seen=self.seen, mark=False)
        refs = self.refs.add(hits)
        blocks: list[str] = []
        for ref, hit, text in zip(refs, hits, texts):
            head = (f"R{ref} · {hit.platform} {hit.partner} · {hit.chunk.day} · "
                    f"{len(hit.lines)} msgs")
            blocks.append(f"{head}\n{text}")
            mark_shown([(hit, text)], self.seen)
            self.injected += _count(text, self.ratio)
        self.calls.append({"tool": "search", "args": {"query": query}, "result_bytes": len(text)})
        return "\n\n".join(blocks)

    def get(self, ref: int, window: int) -> str:
        hit = self.refs.resolve(ref)
        if hit is None:
            return f"no R{ref}. Search first; the refs you can read are the ones you were given."
        if window <= 0:
            texts, _ = render_hits([hit], seen=self.seen, mark=False)
            text = texts[0]
            mark_shown([(hit, text)], self.seen)
            self.injected += _count(text, self.ratio)
            self.calls.append({"tool": "get", "args": {"ref": ref}, "result_bytes": len(text)})
            return text
        msgs = get(self.index, hit.chunk.id, window=window,
                   bundle=self.index.manifest.get("bundle"))
        lines = msgs if isinstance(msgs, list) else []
        fresh = [m for m in lines if (hit.platform, hit.partner, m["i"]) not in self.seen]
        for m in fresh:
            self.seen.add((hit.platform, hit.partner, m["i"]))
        text = "\n".join(m["text"] for m in fresh) or "those messages are already shown above."
        self.injected += _count(text, self.ratio)
        self.calls.append({"tool": "get", "args": {"ref": ref, "window": window},
                           "result_bytes": len(text)})
        return text


def parse(text: str) -> tuple[str, Any] | None:
    """The last command in a message, as (verb, arg). `None` means the model answered in prose."""
    hits = list(CMD.finditer(text))
    if not hits:
        return None
    m = hits[-1]
    verb, rest = m.group(1).lower(), m.group(2).strip().lstrip("*_`:. ").strip()
    if verb == "answer":
        return "answer", text[m.end(2) - len(rest):].strip()
    if verb == "search":
        return "search", rest.strip("\"'` \t")
    got = GET_ARG.search(rest)
    if not got:
        return "get", (0, 0)
    return "get", (int(got.group(1)), int(got.group(2) or 0))


def _post(base_url: str, api_key: str, payload: dict, timeout: float = 300) -> dict:
    import requests

    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      headers={"Authorization": f"Bearer {api_key}"}, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


def run(index: Index, question: str, *, cfg: dict, embedder: Embedder | None = None,
        reranker: Reranker | None = None, prefill: bool = False, pull: bool = True,
        k: int | None = None, dry_run: bool = False) -> LoopResult:
    llm, ret = cfg["llm"], cfg["retrieve"]
    k = k or ret["k"]
    donor = index.manifest.get("donor_id") or index.manifest.get("donor") or "the donor"
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM.format(donor=donor)}]

    state = State(index=index, refs=RefRegistry(), seen=set(), ret=ret, k=k,
                  embedder=embedder, reranker=reranker,
                  ratio=float(index.manifest.get("chars_per_token") or DEFAULT_RATIO))

    if prefill:  # one block covering the probes, before the model has said anything
        p = push_block([question], lambda q, n: search_any(
            index, q, embedder, k=n, reranker=reranker,
            max_region_chunks=ret["max_region_chunks"]),
            k=ret["k"], budget_tokens=ret["budget"], min_threads=ret["min_threads"],
            ratio=state.ratio)
        if p.texts:
            messages.append({"role": "user",
                             "content": "Context from your own chat history:\n\n" + "\n\n".join(p.texts)})
            state.injected += p.stats["tokens"]
            for h in p.hits:
                state.refs.add([h])

    messages.append({"role": "user", "content": question})

    if dry_run:
        return LoopResult(answer="", turns=0, injected_tokens=state.injected,
                          calls=[{"dry_run": {"messages": messages}}])

    nudged = 0
    for turn in range(int(llm["max_turns"])):
        payload = {"model": llm["model"], "messages": messages,
                   "temperature": llm["temperature"], "max_tokens": int(llm.get("max_tokens") or 700)}
        resp = _post(llm["base_url"], llm.get("api_key", "sk-local"), payload)
        content = (resp["choices"][0]["message"].get("content") or "").strip()
        messages.append({"role": "assistant", "content": content})

        cmd = parse(content)
        # A small model asked a survey question will answer from imagination, or copy the example's
        # words. So the first search is not its job: the harness runs the question itself and hands
        # over the result. The model's first real decision is whether that was enough.
        if (cmd is None or cmd[0] == "answer") and nudged == 0 and pull and not state.seen:
            nudged += 1
            messages.append({"role": "user", "content":
                             "Observation (the question, searched verbatim):\n"
                             + state.search(question)})
            continue
        if cmd is None or cmd[0] == "answer":
            answer = content if cmd is None else (cmd[1] or "")
            if not answer.strip() and nudged < 2:
                # "answer" with nothing after it is a formatting failure, not a finding.
                nudged += 1
                messages.append({"role": "user", "content":
                                 "Write the answer itself: `answer <a few sentences>`."})
                continue
            return LoopResult(answer or content, turn + 1, state.calls, state.injected, "answered")

        verb, arg = cmd
        if not pull:  # answer-only mode: a command is a mistake, say so once and take the prose
            messages.append({"role": "user", "content": "You cannot search in this mode. Answer."})
            continue
        if len(state.calls) >= int(llm["max_calls"]):
            return LoopResult(content, turn + 1, state.calls, state.injected, "max_calls")

        if verb == "search":
            observation = state.search(str(arg))
        else:
            ref, window = arg
            observation = state.get(ref, window)
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

        if state.injected > ret["budget"]:
            return LoopResult(content, turn + 1, state.calls, state.injected, "budget")
    return LoopResult("", int(llm["max_turns"]), state.calls, state.injected, "max_turns")
