"""Load a thesis-deid donation bundle into Thread/Msg objects.

Defensive by design: the uploader leaks group threads (see FINDINGS-uploader.md),
so validation is independent of the bundle's own `participants`/`partner` fields.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


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
            rejects.append(Reject(str(path), t.id, "group-leak",
                                  f"{len(senders)} senders {top}"))
        elif not senders:
            rejects.append(Reject(str(path), t.id, "empty", "0 senders"))
        else:
            threads.append(t)
    rp = root / "report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    return threads, rejects, report
