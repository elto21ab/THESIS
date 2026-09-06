"""Shared schema and helpers for the preprocess pipeline.

All sources standardise into `Thread` + `Msg`. Names stay RAW here — SUBJECT/OTHER
relabelling is a de-identification concern, not a parsing one.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

Platform = Literal["fb", "ig", "wa"]
MediaKind = Literal["audio", "photos", "videos", "other"]


@dataclass
class MediaRef:
    uri: str
    kind: MediaKind
    missing: bool = False


@dataclass
class MediaRef:
    uri: str
    kind: MediaKind
    missing: bool = False
    # concrete locator for later byte-copy (--with-media). One of:
    #   {"zip": <archive index>, "entry": <zip entry name>}
    #   {"file": <abs path>}
    loc: dict | None = None
    # original source uri as it appeared in the export (unchanged even after
    # uri is rewritten to the flat-store hash name)
    orig: str = ""


@dataclass
class Reaction:
    emoji: str
    actor: str = ""
    ts: int = 0


@dataclass
class Msg:
    ts: int  # epoch ms
    sender: str
    text: str = ""
    media: list[MediaRef] = field(default_factory=list)
    reactions: list[Reaction] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    system: bool = False
    id: str = ""

    def dedupe_key(self) -> str:
        """Content key: timestamp + sender + first 120 chars of text (folded)."""
        folded = " ".join(self.text.split()).lower()[:120]
        return f"{self.ts}|{self.sender.strip().lower()}|{folded}"


@dataclass
class Thread:
    platform: Platform
    id: str  # platform-scoped: fb001, ig004, wa002
    partner: str = ""
    participants: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)  # e.g. ["fb-inbox", "fb-e2ee-dump"]
    messages: list[Msg] = field(default_factory=list)
    first_ms: int = 0
    last_ms: int = 0
    # where thread media lives, for later byte-copy:
    #   zip: archive index + entry prefix; file: abs dir path
    media_zip: int | None = None
    media_base: str = ""

    def finalise(self) -> None:
        ts = [m.ts for m in self.messages if m.ts > 0]
        if ts:
            self.first_ms, self.last_ms = min(ts), max(ts)


@dataclass
class SourceStats:
    platform: Platform
    raw_threads: int = 0
    group_dropped: int = 0
    messages: int = 0
    text_chars: int = 0
    archives: int = 0
    media: dict[MediaKind, int] = field(default_factory=lambda: {"audio": 0, "photos": 0, "videos": 0, "other": 0})
    escapes_repaired: int = 0


def thread_sig(participants: list[str]) -> str:
    """Normalised identity: lowercased, sorted participant names."""
    return "|".join(sorted(p.strip().lower() for p in participants if p.strip()))


def media_kind_of(uri: str) -> MediaKind:
    n = uri.lower()
    if n.endswith((".opus", ".ogg", ".wav", ".m4a", ".mp3", ".aac", ".amr")):
        return "audio"
    if n.endswith((".jpg", ".jpeg", ".png", ".gif", ".heic", ".webp", ".bmp")):
        return "photos"
    if n.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return "videos"
    if "audio" in n:
        return "audio"
    if "video" in n:
        return "videos"
    if "photo" in n:
        return "photos"
    return "other"


# -- Meta's byte-escaped UTF-8 repair -----------------------------------------
# Messenger/IG export strings arrive as valid UTF-8 JSON whose *content* is often
# double-encoded (UTF-8 bytes re-read as Latin-1). Repair conservatively: when a
# string is all-Latin-1-decodable and re-decoding as UTF-8 removes mojibake tokens
# without introducing replacement chars, take the repaired form.

_MOJIBAKE_TOKENS = ("Ã", "Â", "â€", "Å", "æ’", "â€™", "â€œ", "â€", "ðŸ", "â€¦", "â€“")


def repair_mojibake(s: str) -> tuple[str, int]:
    if not s or not any(t in s for t in _MOJIBAKE_TOKENS):
        return s, 0
    try:
        reparsed = s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s, 0
    if "\ufffd" in reparsed:
        return s, 0
    before = sum(s.count(t) for t in _MOJIBAKE_TOKENS)
    after = sum(reparsed.count(t) for t in _MOJIBAKE_TOKENS)
    if after < before:
        return reparsed, 1
    return s, 0


URL_RE = re.compile(r"https?://[^\s'\"<>]+|www\.[^\s'\"<>]+")


def extract_urls(text: str) -> list[str]:
    """Pull http(s)/www links out of message text (best-effort, no validation)."""
    if not text:
        return []
    out = []
    for m in URL_RE.findall(text):
        u = m.rstrip(",.;:)!?]}")
        if u not in out:
            out.append(u)
    return out


def enrich_msg(m: Msg) -> Msg:
    """Fill derived fields (urls) after parse. Source-agnostic."""
    m.urls = extract_urls(m.text)
    return m


def stable_id(prefix: str, seed: str) -> str:
    return f"{prefix}{hashlib.sha1(seed.encode()).hexdigest()[:8]}"


def iso_ms(ms: int) -> str:
    """Epoch ms → 'YYYYMMDD-HHMMSS' (UTC)."""
    from datetime import datetime, timezone

    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y%m%d-%H%M%S")