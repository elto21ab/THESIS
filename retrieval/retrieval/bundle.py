"""Bundle in, prompt lines out.

`load_bundle` reads a thesis-deid donation bundle into `Thread`/`Msg` objects, defensively: the
uploader leaks group threads (see FINDINGS-uploader.md), so validation keys on **actual senders**,
never on the bundle's own `participants`/`partner` metadata.

`render_msg` produces the one prompt-line format (`[ts] SENDER: text <attached: …> <react: …>
<call: …> <link: …>`), mirroring thesis-deid `minimal.ts`. Never-parsed media collapse to a
**kind mask** (`PHOTO`/`VIDEO`/`FILE`); audio keeps a mask plus its index (SPEC §5) so the STT
backfill can land in the same slot later. Bursts render as `xN`.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


# ── types ─────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class Media:
    name: str
    kind: str
    missing: bool = False


@dataclass(slots=True)
class Msg:
    ts: int  # ms epoch, UTC
    sender: str
    text: str
    urls: list[str] = field(default_factory=list)
    reactions: list[tuple[str, str]] = field(default_factory=list)  # (emoji, actor)
    media: list[Media] = field(default_factory=list)
    call_s: int | None = None
    call_media: str | None = None
    call_missed: bool = False


@dataclass(slots=True)
class Thread:
    id: str
    platform: str
    partner: str
    participants: list[str]
    messages: list[Msg]  # ascending ts
    sources: list[str] = field(default_factory=list)

    def senders(self) -> set[str]:
        return {m.sender for m in self.messages}


@dataclass(slots=True)
class Reject:
    path: str
    id: str
    reason: str
    detail: str


# ── load ──────────────────────────────────────────────────────────────────────
def _parse_msg(d: dict) -> Msg:
    cs = d.get("call_s")
    return Msg(
        ts=int(d.get("ts_ms") or 0),
        sender=str(d.get("sender") or ""),
        text=str(d.get("text") or ""),
        urls=[str(u) for u in d.get("urls") or []],
        reactions=[(str(r.get("emoji", "")), str(r.get("actor", "")))
                   for r in d.get("reactions") or []],
        media=[Media(str(m.get("name", "")), str(m.get("kind", "other")),
                     bool(m.get("missing"))) for m in d.get("media") or []],
        call_s=int(cs) if cs is not None else None,
        call_media=d.get("call_media"),
        call_missed=bool(d.get("call_missed", False)),
    )


def load_thread(path: Path) -> Thread:
    raw = json.loads(path.read_text())
    msgs = [_parse_msg(m) for m in raw.get("messages") or []]
    msgs.sort(key=lambda m: m.ts)  # bundle order is NOT chronological — see PLAN.md
    return Thread(
        id=str(raw.get("id") or path.stem),
        platform=str(raw.get("platform") or path.parent.name),
        partner=str(raw.get("partner") or ""),
        participants=[str(p) for p in raw.get("participants") or []],
        messages=msgs,
        sources=[str(s) for s in raw.get("sources") or []],
    )


def load_bundle(root: str | Path, *, max_senders: int = 2,
                ) -> tuple[list[Thread], list[Reject], dict]:
    """Return (threads, rejects, report). Rejects carry a reason for the log."""
    root = Path(root)
    threads: list[Thread] = []
    rejects: list[Reject] = []
    for path in sorted(root.glob("*/*.json")):
        if path.name == "report.json":
            continue
        try:
            t = load_thread(path)
        except Exception as e:  # noqa: BLE001
            rejects.append(Reject(str(path), path.stem, "unreadable", str(e)))
            continue
        senders = t.senders()
        if not t.partner or t.partner == "SUBJECT":
            rejects.append(Reject(str(path), t.id, "no-partner", f"partner={t.partner!r}"))
        elif len(senders) > max_senders:
            top = Counter(m.sender for m in t.messages).most_common(4)
            rejects.append(Reject(str(path), t.id, "group-leak", f"{len(senders)} senders {top}"))
        elif not senders:
            rejects.append(Reject(str(path), t.id, "empty", "0 senders"))
        else:
            threads.append(t)
    rp = root / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    return threads, rejects, report


def thread_for(root: str | Path, platform: str, partner: str) -> Thread | None:
    """Reload one thread from a bundle (used by `get --window`). ~0.5 s: scans the platform dir."""
    for p in sorted((Path(root) / platform).glob("*.json")):
        if p.name == "report.json":
            continue
        try:
            t = load_thread(p)
        except Exception:  # noqa: BLE001
            continue
        if t.partner == partner and t.platform == platform:
            return t
    return None


# ── render ────────────────────────────────────────────────────────────────────
_COLLAPSE = {"photos": "PHOTO", "videos": "VIDEO"}


def _attach_label(m: Media, i: int) -> str:
    if m.kind == "audio" and m.name:
        return f"AUDIO#{i}"  # mask + slot for the future transcript; the name lives in the bundle
    if m.missing or not m.name:
        return m.name or "FILE"
    return _COLLAPSE.get(m.kind, "FILE")


def render_msg(m: Msg) -> str:
    ts = datetime.fromtimestamp(m.ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {m.sender}: {m.text}"
    if m.media:
        order: list[str] = []
        count: dict[str, int] = {}
        for i, x in enumerate(m.media, 1):
            label = _attach_label(x, i)
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


def logical_day(ts_ms: int, tz: str, start_hour: int) -> str:
    """A day runs `start_hour` → `start_hour` local, so a 02:00 message belongs to the night
    before. Storage stays UTC; only this boundary is local."""
    dt = datetime.fromtimestamp(ts_ms / 1000, ZoneInfo(tz)) - timedelta(hours=start_hour)
    return dt.strftime("%Y-%m-%d")
