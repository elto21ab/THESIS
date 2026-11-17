"""One message as the model reads it — the reference renderer, in Python.

This is a port of `thesis-deid/src/core/standard/minimal.ts`, and it exists for one reason: a bundle
must be re-renderable without the donor. `as_prompt` in the bundle is a cache of this function; when
the format changes, the server renders again from the fields it already holds instead of asking
somebody to upload 38 GB for the third time.

`prep/parity.py` proves the two implementations agree, message by message, over the whole corpus. If
they ever drift, the parity check fails and that is the bug — not a difference of opinion.

The rules, in the order they were decided:

    [2022-09-05 13:59] S: This one on Tuesday seems nice, too. <react: ❤ by O1>
    [14:02] O1: <attached: PHOTO.jpg> <voice 0:12>

    timestamp   `YYYY-MM-DD HH:MM` when the day changes, `HH:MM` otherwise. The date on every message
                repeated what the line above already said, and was 12.5% of the prompt.
    label       `S` for the donor, `O<n>` for `OTHER-nnn` — the shorthand the study declares.
    text        as written, newlines and all.
    audio       `<voice 0:12>` untranscribed, `<voice>…</voice>` once a transcript exists.
    attachment  `<attached: PHOTO.jpg>`, `VIDEO.mp4`, `GIF`, `STICKER`, `FILE.pdf`, `xN` for a burst.
    reactions   `<react: ❤ by S at 11:46>`, `<react: sticker by O3>`.
    calls       `<call: in video 0s missed from O1>`, `<call: event video>`, `<call: 0s>`.
    urls        `<link: …>` only for a link the text does not already spell out.
    unsent      `<unsent>`.
    empty       `<no content>`.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Callable, Iterable, Optional

RENDER_VERSION = "v4"

SHARE_TITLE_MAX = 200

KIND_PLACEHOLDER = {
    "photos": "PHOTO.jpg",
    "videos": "VIDEO.mp4",
    "audio": "VOICE.m4a",
    "gif": "GIF",
    "sticker": "STICKER",
    "file": "FILE",
}

_PAD = lambda n: f"{n:02d}"  # noqa: E731


def _utc(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def stamp_for(ms: int, previous: Optional[int] = None) -> str:
    """The stamp a message carries: full date when the day changes, `HH:MM` otherwise."""
    if not ms:
        return "     "  # same width as `HH:MM`, so a line without a time still lines up
    d = _utc(ms)
    clock = f"{_PAD(d.hour)}:{_PAD(d.minute)}"
    if previous and previous > 0 and _utc(previous).date() == d.date():
        return clock
    return f"{d.year}-{_PAD(d.month)}-{_PAD(d.day)} {clock}"


def duration_text(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole // 60}:{_PAD(whole % 60)}"


def short_label(label: str) -> str:
    """`SUBJECT` → `S`, `OTHER-001` → `O1`."""
    if not label:
        return ""
    if label == "SUBJECT":
        return "S"
    match = re.fullmatch(r"OTHER-(\d+)", label)
    return f"O{int(match.group(1))}" if match else label


def _slice_utf16(text: str, limit: int) -> str:
    """The first `limit` UTF-16 code units of `text` — what JavaScript's `String.slice` means.

    Not pedantry: `SHARE_TITLE_MAX` is 200 units, and a caption of emoji spends two units each, so
    counting code points here cuts later than the donor tool did and the re-render disagrees on 18
    shares. A surrogate pair split by the limit is dropped, which is what JS produces too.
    """
    raw = text.encode("utf-16-le")
    return raw[: limit * 2].decode("utf-16-le", errors="ignore")


def share_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return title
    if len(title.encode("utf-16-le")) // 2 <= SHARE_TITLE_MAX:
        return title
    cut = _slice_utf16(title, SHARE_TITLE_MAX)
    at = cut.rfind(" ")
    return f"{(cut[:at] if at > SHARE_TITLE_MAX / 2 else cut).rstrip()}…"


def _same_day(a: int, b: int) -> bool:
    return _utc(a).date() == _utc(b).date()


def markers_of(message: dict, label_of: Callable[[str], str]) -> str:
    """Everything a message carries besides its words, as the markers that follow the text."""
    parts: list[str] = []
    for reaction in message.get("reactions") or []:
        # A reaction can be a sticker instead of an emoji (Instagram writes an empty emoji plus a
        # second field). Skipping the empty emoji dropped the reaction and its actor entirely.
        mark = reaction.get("emoji") or ("sticker" if reaction.get("sticker") else "")
        if not mark:
            continue
        at = ""
        if reaction.get("ts"):
            when = reaction["ts"]
            at = f" at {_utc(when).strftime('%H:%M') if _same_day(when, message.get('ts', 0)) else stamp_for(when)}"
        actor = f" by {label_of(reaction['actor'])}" if reaction.get("actor") else ""
        parts.append(f"<react: {mark}{actor}{at}>")

    counts: dict[str, int] = {}
    for ref in message.get("media") or []:
        if ref.get("kind") == "audio":
            transcript = ref.get("transcript")
            if transcript:
                parts.append(f"<voice>{transcript}</voice>")
            elif ref.get("duration") is not None:
                parts.append(f"<voice {duration_text(ref['duration'])}>")
            else:
                parts.append("<voice>")
            continue
        # The extension is the only thing left of a document, and it is signal: `.mmap` is a mind map,
        # `.vcf` a contact card. The kind names the class, the suffix names the thing.
        match = re.search(r"\.[A-Za-z0-9]{1,6}$", ref.get("name") or "")
        ext = match.group(0).lower() if match else ""
        name = f"FILE{ext}" if ref.get("kind") == "file" else KIND_PLACEHOLDER.get(ref.get("kind", ""), "FILE")
        counts[name] = counts.get(name, 0) + 1
    for name, count in counts.items():
        parts.append(f"<attached: {name} x{count}>" if count > 1 else f"<attached: {name}>")

    if message.get("call"):
        call = message["call"]
        direction = call.get("direction")
        media = call.get("media", "unknown")
        seconds = call.get("seconds", 0) or 0
        actor = call.get("actor")
        if direction == "event":
            length = f" {seconds}s" if seconds > 0 else ""
            who = f" by {actor}" if actor else ""
            parts.append(f"<call: event{'' if media == 'unknown' else f' {media}'}{length}{who}>")
        else:
            who = f"{' from' if direction == 'in' else ' to' if direction == 'out' else ' by'} {actor}" if actor else ""
            kind = "" if media == "unknown" else f"{media} "
            parts.append(f"<call: {f'{direction} ' if direction else ''}{kind}{seconds}s{' missed' if call.get('missed') else ''}{who}>")

    # A link the message already spells out is already there. Compared without the trailing slash,
    # because Meta stores the reference with one and the author's text rarely has it.
    def trim(url: str) -> str:
        return url.rstrip("/")

    text = message.get("text") or ""
    in_text = trim(text)
    candidates = [in_text] + [trim(u) for u in re.findall(r"https?://[^\s'\"<>]+", text)]

    def seen(url: str) -> bool:
        u = trim(url)
        if u in in_text:
            return True
        # A reference can extend a URL in the text by a suffix (`…/2411.11581v4`): the same resource.
        return any((u.startswith(c) and len(u) - len(c) <= 8) or (c.startswith(u) and len(c) - len(u) <= 8) for c in candidates)

    for url in message.get("urls") or []:
        if not seen(url):
            parts.append(f"<link: {url}>")

    share = message.get("share")
    if share:
        label = " ".join(
            bit for bit in (share_title(share.get("title")), f"by {share['owner']}" if share.get("owner") else "") if bit
        )
        parts.append(f"<share: {label}>" if label else "<share: post>")

    if message.get("unsent"):
        parts.append("<unsent>")
    return " ".join(parts)


def render_message(message: dict, label_of: Callable[[str], str], previous: Optional[int] = None) -> str:
    """One message as the model reads it. Not a line: the text keeps its own newlines."""
    markers = markers_of(message, label_of)
    # Two kinds of line have no speaker: one the platform gave no sender, and a notice, which nobody
    # said even when the export attributes it to them. A deletion keeps its sender.
    label = "" if message.get("system") == "notice" else short_label(label_of(message.get("sender", "")))
    who = f" {label}:" if label else ""
    # A call written as a sentence is the same fact as its marker, so the marker speaks alone.
    prose = "" if (message.get("call") or {}).get("prose") else message.get("text") or ""
    body = f" {prose}" if prose else ""
    nothing = " <no content>" if not body and not markers else ""
    return f"[{stamp_for(message.get('ts', 0), previous)}]{who}{body}{f' {markers}' if markers else ''}{nothing}"


def message_for_render(record: dict) -> dict:
    """A bundle message record, in the shape `render_message` takes.

    The bundle keeps the fields flat (`call_s`, `call_missed`, `call_media`, …) because a consumer
    reads them flat; the renderer wants them nested, because the marker is about a call as a whole.
    """
    call = None
    if "call_s" in record:
        call = {
            "seconds": record.get("call_s", 0),
            "missed": record.get("call_missed", False),
            "media": record.get("call_media", "unknown"),
            # None when the donor tool inferred it: a guess must not be printed as a fact.
            "direction": None if record.get("call_direction_inferred") else record.get("call_direction"),
            # The export's own words when it had them (`Emma`), the label only when that is all there
            # is. The donor tool rendered the raw string, and a re-render must print what it printed.
            **({"actor": record.get("call_actor_raw") or record["call_actor"]} if record.get("call_actor") else {}),
            **({"prose": True} if record.get("call_prose") else {}),
        }
    return {
        "ts": record.get("ts_ms", 0),
        "sender": record.get("sender", ""),
        "text": record.get("text", ""),
        "urls": record.get("urls") or [],
        "media": record.get("media") or [],
        "reactions": record.get("reactions") or [],
        **({"call": call} if call else {}),
        **({"unsent": True} if record.get("unsent") else {}),
        **({"system": record["system"]} if record.get("system") else {}),
        **({"share": record["share"]} if record.get("share") else {}),
    }


def render_thread(messages: Iterable[dict]) -> list[str]:
    """Every message of one conversation, in order."""
    out: list[str] = []
    previous: Optional[int] = None
    for record in messages:
        out.append(render_message(message_for_render(record), lambda raw: raw, previous))
        previous = record.get("ts_ms") or 0
    return out
