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
                  render_hits, search_tool_schema, thread_note, window_lines)
from .search import search_any
from .tokens import DEFAULT_RATIO, count as _count

SYSTEM = """You answer a survey as {donor}, using only messages from {donor}'s own chat history.

You have three commands. Write exactly one per message, on its own line:

  search <words>      search by meaning. Returns numbered regions R1, R2, … with the messages.
  grep <pattern>      find a literal word, name, number or date. Regex, no interpretation.
  get <ref> [±N]      read region R3 again, or N messages around it: `get 3 ±5`
  answer <text>       stop searching and answer the question.

You are {donor} — a person with a life, not an assistant. Never mention being a model, an AI, or a
system; if the history is silent, say the history is silent, in the first person.

Rules that matter:
- Ground every claim in a message you actually read. If the history does not answer the question, say
  so with `answer` instead of inventing.
- Prefer one good region over many shallow ones. Search again only when the last result was beside
  the point.
- Write the answer in the language of the question, first person, a few sentences.

The search words are yours to choose from the question — two or three words the donor would have
typed, not the whole question.
"""

#: FunctionGemma writes tool calls as text even when it is handed a schema:
#: `<start_function_call>call:search{query:<escape>zumba<escape>}<end_function_call>`. llama.cpp does
#: not map that back to OpenAI `tool_calls`, so the harness reads both wire formats — a model that
#: speaks either can drive the same executor.
FG_CALL = re.compile(r"call:([A-Za-z_][\w-]*)\{(.*?)\}", re.DOTALL)


def fg_calls(text: str) -> list[tuple[str, dict]]:
    """`call:search{query:<escape>zumba<escape>}` → [("search", {"query": "zumba"})]."""
    out: list[tuple[str, dict]] = []
    for name, body in FG_CALL.findall(text):
        args: dict[str, Any] = {}
        for m in re.finditer(r"([\w-]+):\s*(<escape>(.*?)<escape>|\{[^{}]*\}|[\w.±+-]+)",
                             body, re.DOTALL):
            key = m.group(1)
            if m.group(3) is not None:
                args[key] = m.group(3)
            elif m.group(2).startswith("{"):
                args[key] = m.group(2)
            else:
                value = m.group(2)
                args[key] = int(value) if value.lstrip("-").isdigit() else value
        out.append((name, args))
    return out


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
    question: str = ""
    shown: list[str] = field(default_factory=list)   # one line per region already read

    def rerank_context(self, query: str) -> str:
        """What the cross-encoder is told, beyond the probe: the survey question, then the regions
        already on the table. A bi-encoder cannot use this (it must pre-compute), a cross-encoder can
        — and a follow-up search reading "more like this, not like what we already have" is the
        difference between a second stage that adds signal and one that re-orders the same six."""
        parts = [f"Survey question: {self.question}", f"Searching for: {query}"]
        if self.shown:
            parts.append("Already read (prefer new evidence): " + "; ".join(self.shown[-5:]))
        return "\n".join(parts)

    def search(self, query: str, mode: str = "dense") -> str:
        hits = search_any(self.index, query, self.embedder, k=self.k,
                          reranker=self.reranker if mode == "dense" else None,
                          mode=mode,
                          max_region_chunks=self.ret["max_region_chunks"],
                          rerank_query=self.rerank_context(query),
                          hybrid_tau=self.ret.get("hybrid_tau"),
                          min_rerank_score=self.ret.get("min_rerank_score"))
        if not hits:
            return "no messages matched. Try fewer words, or a word the donor would have used."
        texts, _ = render_hits(hits, seen=self.seen, mark=False)
        refs = self.refs.add(hits)
        blocks: list[str] = []
        for ref, hit, text in zip(refs, hits, texts):
            note = (thread_note(self.index.manifest, hit.platform, hit.partner)
                    if self.ret.get("thread_stats") else "")
            head = (f"R{ref} · {hit.platform} {hit.partner} · {hit.chunk.day} · "
                    f"{len(hit.lines)} msgs" + (f" · {note}" if note else ""))
            blocks.append(f"{head}\n{text}")
            mark_shown([(hit, text)], self.seen)
            self.shown.append(f"R{ref} {hit.partner} {hit.chunk.day}")
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
            self.shown.append(f"R{ref} {hit.partner} {hit.chunk.day}")
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


FG_ANSWER = re.compile(r"call:answer\{text:\s*<escape>(.*?)<escape>", re.DOTALL)


def fg_answer(text: str) -> str | None:
    """The last `call:answer{text:<escape>…<escape>}` in the message, however badly it is closed."""
    found = FG_ANSWER.findall(text)
    return clean(found[-1]) if found else None


def clean(text: str) -> str:
    """Strip the model's own call markup out of an answer. A 270M model will paste it in."""
    text = re.sub(r"</?(?:start|end)_function_(?:call|response)>", "", text)
    text = re.sub(r"<escape>|</?escape>", "", text)
    text = re.sub(r"call:\w+\{", "", text)
    text = text.replace("text:", "", 1) if text.lstrip().startswith("text:") else text
    text = re.sub(r"\}?\s*</?(?:start|end)_function_(?:call|response)>.*$", "", text, flags=re.DOTALL)
    return text.strip(" ,}\n")


def _schemas(reg: RefRegistry) -> list[dict]:
    """`search` always, `get` only once there is a ref to name (its enum is the set of refs the model
    was actually handed, so it cannot invent one), and `answer` — without it a tool-tuned model has
    no way to *finish*, and a 270M one will happily `get 1..26` until the call cap stops it."""
    out = [search_tool_schema(),
           {"type": "function", "function": {
               "name": "grep",
               "description": "find a literal string — a name, a number, a date, a quoted phrase",
               "parameters": {"type": "object",
                              "properties": {"pattern": {"type": "string"}},
                              "required": ["pattern"]}}}]
    if len(reg):
        out.append(reg.get_tool_schema())
    out.append({"type": "function", "function": {
        "name": "answer",
        "description": "stop searching and answer the survey question",
        "parameters": {"type": "object",
                       "properties": {"text": {"type": "string", "description": "the answer"}},
                       "required": ["text"]}}})
    return out


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
                  embedder=embedder, reranker=reranker, question=question,
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
        payload: dict[str, Any] = {"model": llm["model"], "messages": messages,
                                   "temperature": llm["temperature"],
                                   "max_tokens": int(llm.get("max_tokens") or 700)}
        if llm.get("tools") and pull:
            payload["tools"] = _schemas(state.refs)
            payload["tool_choice"] = "auto"
        resp = _post(llm["base_url"], llm.get("api_key", "sk-local"), payload)
        content = (resp["choices"][0]["message"].get("content") or "").strip()
        messages.append({"role": "assistant", "content": content})

        # Three wire formats, one executor: OpenAI tool_calls (llama.cpp / vLLM with --jinja),
        # FunctionGemma's own `call:name{…}`, or a plain-text command line.
        calls: list[tuple[str, dict]] = []
        for call in (resp["choices"][0]["message"].get("tool_calls") or []):
            fn = call.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append((fn.get("name", ""), args if isinstance(args, dict) else {}))
        if not calls:  # FunctionGemma: several calls, and it often restates the answer before it settles
            calls = fg_calls(content)

        if not calls:
            tail = fg_answer(content)
            if tail:
                return LoopResult(tail, turn + 1, state.calls, state.injected, "answered")
        cmd = None if calls else parse(content)
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
            answer = clean(content) if cmd is None else (cmd[1] or "")
            if not answer.strip() and nudged < 2:
                # "answer" with nothing after it is a formatting failure, not a finding.
                nudged += 1
                messages.append({"role": "user", "content":
                                 "Write the answer itself: `answer <a few sentences>`."})
                continue
            return LoopResult(answer or content, turn + 1, state.calls, state.injected, "answered")

        if calls:
            if not pull:
                messages.append({"role": "user", "content": "You cannot search in this mode. Answer."})
                continue
            if len(state.calls) >= int(llm["max_calls"]):
                return LoopResult(content, turn + 1, state.calls, state.injected, "max_calls")
            answers = [str(k.get("text") or "") for n, k in calls if n == "answer"]
            if answers:
                return LoopResult(clean(answers[-1]), turn + 1, state.calls, state.injected,
                                  "answered")
            observations = []
            for name, kwargs in calls[:4]:
                if name == "search":
                    observations.append(state.search(str(kwargs.get("query", "")), "dense"))
                elif name == "grep":
                    observations.append(state.search(str(kwargs.get("pattern", "")), "lexical"))
                elif name == "get":
                    observations.append(state.get(int(kwargs.get("ref") or 0),
                                                  int(kwargs.get("window") or 0)))
                else:
                    observations.append(f"unknown tool {name!r}. Use search, grep or get.")
            messages.append({"role": "user",
                             "content": "Observation:\n" + "\n\n".join(observations)})
            if turn + 1 >= int(llm["max_turns"]) - 1:
                messages.append({"role": "user", "content":
                                 "Enough searching. Call `answer` with a few sentences."})
            if state.injected > ret["budget"]:
                return LoopResult(content, turn + 1, state.calls, state.injected, "budget")
            continue

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

        if state.injected > int(llm.get("context_limit") or 40_000):
            return LoopResult(content, turn + 1, state.calls, state.injected, "context_limit")
    return LoopResult("", int(llm["max_turns"]), state.calls, state.injected, "max_turns")
