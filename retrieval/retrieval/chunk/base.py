"""Chunk dataclass + config + shared construction. Strategies live beside this file."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

from ..chunk_id import chunk_id
from ..io import Thread
from ..render import thread_header

Count = Callable[[str], int]


@dataclass(slots=True)
class ChunkCfg:
    target: int = 512          # preferred tokens per chunk
    min: int = 256             # merge below this
    max: int = 1024            # hard ceiling
    overlap: float = 0.40      # fraction of target re-included between neighbours
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

    def to_json(self) -> dict:
        return asdict(self)


def logical_day(ts_ms: int, tz: str, start_hour: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, ZoneInfo(tz)) - timedelta(hours=start_hour)
    return dt.strftime("%Y-%m-%d")


def make_chunk(thread: Thread, seq: int, i0: int, i1: int,
               lines: list[str], toks: list[int], cfg: ChunkCfg) -> Chunk:
    day = logical_day(thread.messages[i0].ts, cfg.tz, cfg.day_start_hour)
    header = thread_header(thread.platform, thread.partner, thread.participants, day)
    ts0, ts1 = thread.messages[i0].ts, thread.messages[i1 - 1].ts
    return Chunk(
        id=chunk_id(thread.platform, thread.partner, i0, i1, ts0, ts1),
        platform=thread.platform, partner=thread.partner, seq=seq,
        i0=i0, i1=i1, ts0=ts0, ts1=ts1, n_msgs=i1 - i0,
        n_tokens=sum(toks[i0:i1]), day=day, header=header,
        text=header + "\n" + "\n".join(lines[i0:i1]),
    )


def prepare(thread: Thread, count: Count, cfg: ChunkCfg,
            ) -> tuple[list[str], list[int], list[str], list[str]]:
    """Render a thread once: lines, per-line tokens, per-msg sender, logical day."""
    lines = []
    toks = []
    senders = []
    days = []
    from ..render import render_msg
    for m in thread.messages:
        line = render_msg(m)
        lines.append(line)
        toks.append(count(line))
        senders.append(m.sender)
        days.append(logical_day(m.ts, cfg.tz, cfg.day_start_hour))
    return lines, toks, senders, days
