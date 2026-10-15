"""Chunking: thread → `Chunk[]`, plus the stable chunk id.

Two strategies behind one call (`fixed`, `day`); `day` delegates oversized days to `fixed`, so
`fixed` is both the fallback mechanism and the plain baseline. Chunk records are self-contained
(header line + rendered messages) so the same text is embedded, retrieved, shown, and (if ever
needed) re-embedded without reassembly.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Callable

from .bundle import Thread, logical_day, render_msg, thread_header

Count = Callable[[str], int]

# Bump when the rendered line format or the id inputs change — it invalidates every id.
# v2: audio renders as `AUDIO#k` (was the raw filename); id drops ts0/ts1 (they were decorative:
# a hash only sees its own inputs, so they could never detect a re-sorted bundle).
RENDER_VERSION = "v2"


def chunk_id(platform: str, partner: str, i0: int, i1: int) -> str:
    """Span-addressed, not content-addressed: editing a message keeps the id, so an STT backfill
    re-embeds the same record in place. Stable across re-embeds and re-runs."""
    key = f"{RENDER_VERSION}|{platform}|{partner}|{i0}|{i1}"
    return hashlib.blake2b(key.encode(), digest_size=8).hexdigest()


@dataclass(slots=True)
class ChunkCfg:
    target: int = 512          # preferred tokens per chunk
    min: int = 256             # merge below this
    max: int = 1024            # hard ceiling
    overlap: float = 0.40      # fraction of target re-included between neighbours (recall)
    tz: str = "Europe/Copenhagen"
    day_start_hour: int = 5    # logical day starts 05:00 local
    both_people: bool = True   # soft preference: try to include both senders


@dataclass(slots=True)
class Chunk:
    id: str
    platform: str
    partner: str
    seq: int
    i0: int
    i1: int
    ts0: int
    ts1: int
    n_msgs: int
    n_tokens: int
    day: str
    header: str
    text: str
    line_offsets: list[int] | None = None  # char start of each message line inside `text`

    def to_json(self) -> dict:
        return asdict(self)

    def msg_lines(self) -> list[str]:
        """One rendered line per message, sliced exactly. A message's own text may contain
        newlines, so `text.split("\\n")` is NOT a safe mapping — hence the offsets. Records without
        offsets fall back to the naive split (only correct for single-line messages)."""
        if not self.line_offsets:
            return self.text.split("\n")[1:][: self.i1 - self.i0]
        starts = list(self.line_offsets) + [len(self.text)]
        return [self.text[starts[k]:starts[k + 1]].rstrip("\n")
                for k in range(len(self.line_offsets))]


def make_chunk(thread: Thread, seq: int, i0: int, i1: int,
               lines: list[str], toks: list[int], cfg: ChunkCfg) -> Chunk:
    day = logical_day(thread.messages[i0].ts, cfg.tz, cfg.day_start_hour)
    header = thread_header(thread.platform, thread.partner, thread.participants, day)
    text = header + "\n"
    offsets: list[int] = []
    for ln in lines[i0:i1]:
        offsets.append(len(text))
        text += ln + "\n"
    return Chunk(
        id=chunk_id(thread.platform, thread.partner, i0, i1),
        platform=thread.platform, partner=thread.partner, seq=seq,
        i0=i0, i1=i1, ts0=thread.messages[i0].ts, ts1=thread.messages[i1 - 1].ts,
        n_msgs=i1 - i0, n_tokens=sum(toks[i0:i1]), day=day, header=header,
        text=text.rstrip("\n"), line_offsets=offsets,
    )


def prepare(thread: Thread, count: Count, cfg: ChunkCfg) -> tuple[list[str], list[int], list[str], list[str]]:
    """Render a thread once: lines, per-line tokens, per-msg sender, logical day."""
    lines, toks, senders, days = [], [], [], []
    for m in thread.messages:
        line = render_msg(m)
        lines.append(line)
        toks.append(count(line))
        senders.append(m.sender)
        days.append(logical_day(m.ts, cfg.tz, cfg.day_start_hour))
    return lines, toks, senders, days


# ── fixed: token-budget window ────────────────────────────────────────────────
def chunk_fixed(thread: Thread, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    """Segmentation advances non-overlapping; overlap is added only as a text prefix from the
    previous segment, so a hard day/max boundary cannot re-trigger and cascade tiny duplicates.
    Boundaries snap to message + logical day; both-people is a soft extend within the day."""
    lines, toks, senders, days = prepare(thread, count, cfg)
    n = len(lines)
    chunks: list[Chunk] = []
    start = seq = 0
    while start < n:
        end, tot, who, day = start, 0, set(), days[start]
        while end < n:
            if end > start and days[end] != day and tot >= cfg.min:
                break  # day boundary, once the segment carries signal
            if end > start and tot + toks[end] > cfg.max:
                break
            tot += toks[end]
            who.add(senders[end])
            day = days[end]
            end += 1
            if tot >= cfg.target and (not cfg.both_people or len(who) >= 2):
                break
        if cfg.both_people and len(who) < 2 and end < n:
            while end < n and days[end] == day and tot + toks[end] <= cfg.max:
                tot += toks[end]
                who.add(senders[end])
                end += 1
                if len(who) >= 2:
                    break
        k, back = start, 0.0
        # `max` is a hard ceiling: the overlap may only use the headroom left, and a single
        # backward message that doesn't fit is skipped rather than blowing the budget.
        head = min(cfg.overlap * tot, max(0, cfg.max - tot))
        while k > 0 and toks[k - 1] <= head - back:
            k -= 1
            back += toks[k]
        chunks.append(make_chunk(thread, seq, k, end, lines, toks, cfg))
        seq += 1
        start = end
    return chunks


# ── day: group by logical day, then mend the edges ────────────────────────────
def _day_groups(days: list[str]) -> list[list[int]]:
    out: list[list[int]] = []
    for idx, d in enumerate(days):
        if out and days[out[-1][-1]] == d:
            out[-1].append(idx)
        else:
            out.append([idx])
    return out


def chunk_day(thread: Thread, cfg: ChunkCfg, count: Count) -> list[Chunk]:
    """Group by logical day. Undersized days merge into the nearest neighbour (repeat until
    >= min); oversized days are handed to the fixed window, which respects max."""
    lines, toks, _senders, days = prepare(thread, count, cfg)

    def size(g: list[int]) -> int:
        return sum(toks[i] for i in g)

    groups = _day_groups(days)
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for g in range(len(groups)):
            if size(groups[g]) >= cfg.min:
                continue
            pick = g - 1 if g > 0 else g + 1
            if pick < 0:
                continue
            lo, hi = sorted((g, pick))
            groups[lo:hi + 1] = [sorted(groups[lo] + groups[hi])]
            merged = True
            break

    chunks: list[Chunk] = []
    seq = 0
    for g in groups:
        if size(g) <= cfg.max:
            chunks.append(make_chunk(thread, seq, g[0], g[-1] + 1, lines, toks, cfg))
            seq += 1
            continue
        sub = Thread(thread.id, thread.platform, thread.partner, thread.participants,
                     thread.messages[g[0]:g[-1] + 1], thread.sources)
        for c in chunk_fixed(sub, cfg, count):
            chunks.append(make_chunk(thread, seq, g[0] + c.i0, g[0] + c.i1, lines, toks, cfg))
            seq += 1
    return chunks


CHUNKERS = {"fixed": chunk_fixed, "day": chunk_day}
