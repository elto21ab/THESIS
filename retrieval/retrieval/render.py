"""Msg -> one prompt line, matching thesis-deid `minimal.ts` (single source of truth).

    [YYYY-MM-DD HH:MM:SS] LABEL: text <attached: …> <react: …> <call: …> <link: …>

Never-parsed media collapse to PHOTO/VIDEO/FILE; audio keeps its own name (the
transcript will replace that slot post-STT). Bursts render as `xN`.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .io import Msg

_COLLAPSE = {"photos": "PHOTO", "videos": "VIDEO"}


def _attach_label(m) -> str:
    if m.kind == "audio" or m.missing or not m.name:
        return m.name or "FILE"
    return _COLLAPSE.get(m.kind, "FILE")


def render_msg(m: Msg) -> str:
    ts = datetime.fromtimestamp(m.ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {m.sender}: {m.text}"
    if m.media:
        order: list[str] = []
        count: dict[str, int] = {}
        for x in m.media:
            label = _attach_label(x)
            if label not in count:
                order.append(label)
                count[label] = 0
            count[label] += 1
        line += " " + " ".join(
            f"<attached: {x}{'' if count[x] == 1 else f' x{count[x]}'}>" for x in order)
    for emoji, actor in m.reactions:
        line += f" <react: {emoji} by {actor}>"
    if m.call_s is not None:
        kind = f"{m.call_media} " if m.call_media in ("video", "voice") else ""
        line += f" <call: {kind}call {m.call_s}s{' missed' if m.call_missed else ''}>"
    for u in m.urls:
        line += f" <link: {u}>"
    return line


def thread_header(platform: str, partner: str, participants: list[str], day: str) -> str:
    who = "+".join(participants) if participants else partner
    return f"# {platform} {partner} · {day} · {who}"
