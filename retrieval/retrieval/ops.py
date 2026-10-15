"""The tool surface: push (`pack`), pull (`get`), payloads, and refs for the model.

State (seen messages, running token budget, turn count) lives in the caller's loop, not here, so the
same functions back an in-process loop and a one-shot CLI call. Refs are the exception that needs
per-run state — `RefRegistry` is that state, owned by the loop.

Message identity inside a pack is `(platform, partner, message index)`: exact, no text matching, and
it is what lets overlapping regions be merged once instead of presented twice.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .bundle import Msg, render_msg, thread_for
from .index import Index
from .search import Hit

DEFAULT_MAX_CHARS = 1200
MsgKey = tuple[str, str, int]
_LINE = re.compile(r"^\[(?P<ts>[^\]]+)\] (?P<sender>[^:]+):")


# ── pull ──────────────────────────────────────────────────────────────────────
def get(index: Index, chunk_id: str, *, window: int = 0,
        bundle: str | Path | None = None) -> "object":
    """`window=0` → the chunk. `window>0` → ±window raw rendered messages around it (needs the
    source bundle; falls back to `manifest["bundle"]`)."""
    c = next((c for c in index.chunks if c.id == chunk_id), None)
    if c is None or window <= 0:
        return c
    root = Path(bundle or index.manifest.get("bundle") or "")
    if not root.name:
        raise ValueError("get(window>0) needs --bundle (or a manifest with `bundle`)")
    t = thread_for(root, c.platform, c.partner)
    if t is None:
        raise ValueError(f"thread not found in bundle: {c.platform}/{c.partner}")
    lo, hi = max(0, c.i0 - window), min(len(t.messages), c.i1 + window)
    return [{"i": i, "sender": t.messages[i].sender, "ts": t.messages[i].ts,
             "text": render_msg(t.messages[i])} for i in range(lo, hi)]


# ── push ──────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Pack:
    hits: list[Hit]
    texts: list[str]        # what the model is shown (repeated messages folded to one line)
    stats: dict


def pack(probes: Sequence[str], hits_for: Callable[[str, int], list[Hit]], *,
         k: int = 4, budget_tokens: int = 40_000, min_threads: int = 2,
         dedupe: bool = True) -> Pack:
    """Default injection: for each probe take its hits (already k-limited, filtered and merged by
    the retriever), keep those that contribute at least one *new* message, stop at the token
    budget. The budget is soft — the best hit is always kept, because a survey question that
    retrieves nothing is worse than a small overrun (`budget_hit` flags it). Then top up until
    `min_threads` distinct partners are represented: a persona grounded in one relationship is a
    bad persona."""
    seen: set[MsgKey] = set()
    out: list[Hit] = []
    used = 0
    per_probe: list[int] = []
    truncated = False

    for q in probes:
        n = 0
        for h in hits_for(q, k):
            if not _new_keys(h, seen):
                continue                      # fully covered by what we already have
            if out and used + h.n_tokens > budget_tokens:
                truncated = True
                break
            out.append(h)
            seen |= _keys(h)
            used += h.n_tokens
            n += 1
        per_probe.append(n)
        if truncated:
            break

    threads = {(h.platform, h.partner) for h in out}
    if not truncated and len(threads) < min_threads:
        for q in probes:
            for h in hits_for(q, k * 4):
                if not _new_keys(h, seen):
                    continue
                if used + h.n_tokens > budget_tokens:
                    truncated = True
                    break
                out.append(h)
                seen |= _keys(h)
                used += h.n_tokens
                threads.add((h.platform, h.partner))
                if len(threads) >= min_threads:
                    break
            if truncated or len(threads) >= min_threads:
                break

    texts, fold = render_hits(out, dedupe=dedupe)
    stats = {"n_hits": len(out), "n_msgs": sum(len(h.lines) for h in out), "tokens": used,
             "n_probes": len(probes), "per_probe_new": per_probe, "budget_hit": truncated,
             "threads": len(threads), "min_threads": min_threads, **fold}
    return Pack(out, texts, stats)


def _keys(h: Hit) -> set[MsgKey]:
    return {(h.platform, h.partner, i) for i in h.lines}


def _new_keys(h: Hit, seen: set[MsgKey]) -> list[MsgKey]:
    return [k for k in _keys(h) if k not in seen]


def mark_shown(pairs: Iterable[tuple[Hit, str]], seen: set[MsgKey]) -> int:
    """Record as 'seen' only the messages whose line is actually present in the text returned to the
    model. Truncation and folding mean 'rendered' ≠ 'shown'; marking the wrong one strands messages
    the model never read (a later `get` would return an empty region)."""
    marked = 0
    for h, shown in pairs:
        visible = set(shown.split("\n"))
        for i in h.lines:
            if h.lines[i] in visible and (h.platform, h.partner, i) not in seen:
                seen.add((h.platform, h.partner, i))
                marked += 1
    return marked


def render_hits(hits: Sequence[Hit], *, dedupe: bool = True,
                seen: set[MsgKey] | None = None, mark: bool = True) -> tuple[list[str], dict]:
    """One text per hit for the prompt. With `dedupe`, a message already shown by an earlier hit is
    replaced by a single marker — this is where the 40% overlap between neighbours stops costing
    tokens. Chunk records, ids and vectors are untouched: only what the model reads changes."""
    seen = set() if seen is None else seen   # pass one in to fold across turns, not just within
    # `seen` is only consulted for folding; callers that truncate afterwards pass mark=False and
    # use mark_shown() on the final text instead.
    texts: list[str] = []
    folded = folded_chars = 0
    for h in hits:
        head = min(h.members, key=lambda c: c.i0).header
        kept: list[str] = []
        dropped = 0
        for i in sorted(h.lines):
            key = (h.platform, h.partner, i)
            line = h.lines[i]
            if dedupe and key in seen:
                dropped += 1
                folded += 1
                folded_chars += len(line) + 1
                continue
            if mark:
                seen.add(key)
            if dropped:
                kept.append(f"<… {dropped} msgs shown above …>")
                dropped = 0
            kept.append(line)
        if dropped:
            kept.append(f"<… {dropped} msgs shown above …>")
        texts.append(head + "\n" + "\n".join(kept))
    return texts, {"lines_deduped": folded, "chars_saved": folded_chars,
                   "tokens_saved_est": folded_chars // 4}


# ── payloads (one shape for tool results and CLI output) ──────────────────────
def msg_index(partner: str, lines: dict[int, str], shown: str) -> list[dict]:
    """Per-message metadata for the messages actually *visible* in the returned text, so a specific
    message is citable: `key` = "OTHER-001:793", which joins straight back to the bundle for
    evidence attribution. Parsed from the rendered line (no extra storage, no bundle reload) and
    limited to lines present in `shown` — so truncation and folding keep the index honest, and the
    payload cannot grow beyond what the model can see."""
    visible = set(shown.split("\n"))
    out: list[dict] = []
    for i in sorted(lines):
        if lines[i] not in visible:
            continue
        m = _LINE.match(lines[i])
        out.append({"key": f"{partner}:{i}", "i": i,
                    "ts": m.group("ts") if m else None,
                    "sender": m.group("sender") if m else None})
    return out


def hit_payload(h: Hit, *, max_chars: int = DEFAULT_MAX_CHARS, rank: int | None = None,
                text: str | None = None) -> dict:
    full = h.text
    shown = (full if text is None else text)
    if max_chars > 0:
        shown = shown[:max_chars]
    ts0 = min(c.ts0 for c in h.members)
    ts1 = max(c.ts1 for c in h.members)
    d = {"id": h.chunk.id, "ids": h.ids, "platform": h.platform, "partner": h.partner,
         "day": h.chunk.day, "seq": h.chunk.seq, "i0": h.i0, "i1": h.i1,
         "ts0": ts0, "ts1": ts1, "n_msgs": h.n_msgs, "n_tokens": h.n_tokens,
         "score": round(h.score, 4), "text": shown, "truncated": len(shown) < len(full),
         "msgs": msg_index(h.partner, h.lines, shown)}
    if len(h.members) > 1:
        d["merged"] = len(h.members)
    if rank is not None:
        d["rank"] = rank
    return d


def chunk_payload(c, *, max_chars: int = DEFAULT_MAX_CHARS, text: str | None = None) -> dict:
    """For `get`: a single chunk, flat."""
    full = c.text
    shown = (full if text is None else text)
    if max_chars > 0:
        shown = shown[:max_chars]
    return {"id": c.id, "platform": c.platform, "partner": c.partner, "day": c.day, "seq": c.seq,
            "i0": c.i0, "i1": c.i1, "ts0": c.ts0, "ts1": c.ts1, "n_msgs": c.n_msgs,
            "n_tokens": c.n_tokens, "text": shown, "truncated": len(shown) < len(full),
            "msgs": msg_index(c.partner, {i: ln for i, ln in
                                          zip(range(c.i0, c.i1), c.msg_lines())}, shown)}


def hits_payload(hits: Iterable[Hit], *, max_chars: int = DEFAULT_MAX_CHARS,
                 dedupe: bool = True) -> list[dict]:
    """Same rendered shape as a pack: a message already shown by an earlier hit is folded to one
    marker, so adjacent regions (which share their overlap by construction) cost one copy."""
    hits = list(hits)
    texts, _ = render_hits(hits, dedupe=dedupe)
    return [hit_payload(h, max_chars=max_chars, rank=i, text=t)
            for i, (h, t) in enumerate(zip(hits, texts), 1)]


# ── refs: the model's handle on a hit ─────────────────────────────────────────
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Semantic/regex search over the donor's own chat history. Returns ranked "
                       "regions with a `rank` you can pass to `get`.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "k": {"type": "integer", "default": 8}},
                       "required": ["query"]},
    },
}


def get_tool_schema(refs: Sequence[int]) -> dict:
    """The `get` tool with `ref` constrained to the refs the model has actually seen: grammar /
    constrained decoding makes an unseen value unemittable — impossible, not merely unlikely.
    Integers, not ids: shortest tokens, trivially small logit mask, tokenizer-agnostic."""
    return {
        "type": "function",
        "function": {
            "name": "get",
            "description": "Fetch a chunk you already found. `ref` is the number shown as `rank` in "
                           "a search result. window>0 returns ±window raw messages around it.",
            "parameters": {
                "type": "object",
                "properties": {"ref": {"type": "integer", "enum": sorted(set(refs))},
                               "window": {"type": "integer", "default": 0}},
                "required": ["ref"],
            },
        },
    }


def search_tool_schema() -> dict:
    return SEARCH_TOOL


class RefRegistry:
    """ref (monotonic int) ⇄ hit for one agent run. The loop owns it; the CLI never sees it.
    Refs are deliberately unstable across runs — that is what makes them cheap for the model.
    Nothing outside the registry is selectable, so a stale ref simply cannot be emitted."""

    def __init__(self) -> None:
        self._by_ref: dict[int, Hit] = {}
        self._ref_by_ids: dict[tuple[str, ...], int] = {}
        self._next = 1

    def add(self, hits: Iterable[Hit]) -> list[int]:
        """Register hits and return their refs; a repeated region keeps its ref."""
        refs: list[int] = []
        for h in hits:
            key = tuple(h.ids)
            if key in self._ref_by_ids:
                refs.append(self._ref_by_ids[key])
                continue
            ref = self._next
            self._next += 1
            self._by_ref[ref] = h
            self._ref_by_ids[key] = ref
            refs.append(ref)
        return refs

    def resolve(self, ref: int) -> Hit | None:
        return self._by_ref.get(int(ref))

    def refs(self) -> list[int]:
        return sorted(self._by_ref)

    def get_tool_schema(self) -> dict:
        return get_tool_schema(self.refs())

    def __len__(self) -> int:
        return len(self._by_ref)


def window_lines(msgs: Sequence[Msg]) -> list[str]:
    return [render_msg(m) for m in msgs]
