"""Bundle in, prompt text out.

`load_bundle` reads a thesis-deid donation bundle into `Thread`/`Msg` objects, defensively: the
uploader leaks group threads (see FINDINGS-uploader.md), so validation keys on **actual senders**,
never on the bundle's own `participants`/`partner` metadata.

**This side no longer renders.** Since `RENDER_VERSION` v3 the bundle carries each message as the
model reads it, in `messages[].as_prompt`, written by the one renderer in `thesis-deid`
(`src/core/standard/minimal.ts`). Keeping a second implementation here is what let the two drift:
this repo was still producing v2 text for a v3 bundle, so the corpus was indexed in a format the
report no longer described. A bundle from before that field exists still loads — the text falls back
to `[ts] sender: text` with no markers — and `load_bundle` says so in `report["legacy_render"]`.
"""
from __future__ import annotations

import json
import re
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
    #: The message as the model reads it — stamp, label, text, markers. Absent in a pre-v3 bundle.
    as_prompt: str = ""
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
        as_prompt=str(d.get("as_prompt") or ""),
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
    # Which rendering this bundle was built with, and whether we had to fall back for it.
    report["render_version"] = report.get("render_version")
    legacy = sum(1 for t in threads for m in t.messages if not m.as_prompt)
    if legacy:
        report["legacy_render"] = {
            "messages": legacy,
            "note": "no as_prompt in the bundle: chunk text falls back to `[ts] sender: text` and "
                    "carries no markers. Re-export to get the rendered format.",
        }
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


# ── derived from the bundle, not rendered here ───────────────────────────────
_SHORT_STAMP = re.compile(r"^\[\d{2}:\d{2}\]")


def restamp(line: str, ts: int) -> str:
    """Put the date on a line that only carries a clock.

    The renderer prints `HH:MM` when the message above shares its day, and the full date when the day
    changes — so a message that begins a chunk is rendered short, because the renderer was reading the
    whole thread and could not know where the chunk would start. A chunk must be self-contained, so
    the first message of every chunk gets its date back here: `[14:02] O1: …` → `[2022-09-05 14:02]
    O1: …`. Only the bracket is touched; the body is the renderer's.
    """
    if not _SHORT_STAMP.match(line):
        return line
    stamp = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"[{stamp}]{line[7:]}"
def prompt_of(m: Msg) -> str:
    """The text the model reads for one message.

    `as_prompt` when the bundle carries it. A bundle built before v3 has none, and re-deriving the
    full format here would put the second renderer back; the fallback is the plainest thing that is
    still true — a timestamp and who said it.
    """
    if m.as_prompt:
        return m.as_prompt
    stamp = datetime.fromtimestamp(m.ts / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    return f"[{stamp}] {m.sender}: {m.text}"


def thread_header(platform: str, partner: str) -> str:
    """The one line above a chunk: which channel, which conversation.

    Nothing else belongs here. The date is on the chunk's first message (`chunk.make_chunk` puts it
    back), the participants are `S` and the partner by construction — a thread with three speakers is
    dropped before it is chunked — and the shorthand `fb`/`ig`/`wa`, `S`, `O<n>` is declared once in
    the study's system prompt. `· 2013-06-26 · SUBJECT+OTHER-050` cost 34 characters on 4,646 chunks
    to repeat three things the reader already had.
    """
    return f"# {platform} {partner}"


def logical_day(ts_ms: int, tz: str, start_hour: int) -> str:
    """A day runs `start_hour` → `start_hour` local, so a 02:00 message belongs to the night
    before. Storage stays UTC; only this boundary is local."""
    dt = datetime.fromtimestamp(ts_ms / 1000, ZoneInfo(tz)) - timedelta(hours=start_hour)
    return dt.strftime("%Y-%m-%d")
