"""WhatsApp export reader — per-chat zip with _chat.txt.

Format (ISO-ish, locale-dependent): `[DD/MM/YYYY, HH.MM.SS] Sender: text`
  - continuation lines (no timestamp prefix) append to the previous message
  - media inline: `<attached: 00000025-PHOTO-….jpg>`
  - system lines: "end-to-end encrypted", "deleted this message",
    "Security code changed", missed-call notes -> flagged system.
Timestamps are device-local, timezone-naive; parsed as UTC so they order within the file.
Cross-platform ts comparison is meaningless for WA by design.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

from format import MediaRef, Msg, SourceStats, Thread, media_kind_of

LINE_RE = re.compile(
    r"^[\u200e\u200f\u200b]*\[(\d{2})/(\d{2})/(\d{4}), (\d{2})\.(\d{2})\.(\d{2})\] ([^:]+): ?(.*)$"
)
ATTACH_RE = re.compile(r"<attached:\s*([^>]+)>")
SYSTEM_RE = re.compile(
    r"end-to-end encrypted|deleted this message|security code|missed (voice|video) call|"
    r"changed the (group|security)|created group|added you|removed you|"
    r"joined using this group|left$"
)


def read_whatsapp_zip(path: str | Path) -> tuple[Thread | None, SourceStats]:
    """path: the _chat zip, a folder containing it, or the .txt directly."""
    stats = SourceStats("wa", archives=1)
    p = Path(path)
    # folder → find the whatsapp zip inside; txt → treat as chat text
    zip_path = p
    if p.is_dir():
        cand = next((x for x in p.iterdir() if x.suffix.lower() == ".zip"), None)
        if not cand:
            cand = next((x for x in p.iterdir() if x.name == "_chat.txt"), None)
        if not cand:
            raise ValueError(f"no whatsapp zip or _chat.txt in {p}")
        zip_path = cand

    if zip_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(zip_path) as z:
            txt_name = next((n for n in z.namelist() if n.endswith("_chat.txt")), None)
            if not txt_name:
                raise ValueError(f"whatsapp zip has no _chat.txt: {zip_path.name}")
            raw = z.read(txt_name).decode("utf-8", errors="replace")
            media_files = set(n for n in z.namelist() if not n.endswith("/"))
    else:  # .txt
        raw = zip_path.read_text(encoding="utf-8", errors="replace")
        media_files = set()

    messages: list[Msg] = []
    cur: Msg | None = None
    senders: dict[str, int] = {}

    for line in raw.splitlines():
        m = LINE_RE.match(line)
        if m:
            dd, mm, yyyy, hh, mi, ss, sender_raw, body = m.groups()
            import datetime as _dt

            ts = int(
                _dt.datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss), tzinfo=_dt.timezone.utc).timestamp() * 1000
            )
            sender = sender_raw.strip()
            body = body.replace("\u200b", "").replace("\u200e", "").strip()
            system = bool(SYSTEM_RE.search(body))
            msg = Msg(ts=ts, sender=sender, text=body, system=system)
            for fn in ATTACH_RE.findall(body):
                fn = fn.strip()
                missing = fn not in media_files
                msg.media.append(MediaRef(uri=fn, kind=media_kind_of(fn), missing=missing,
                                          loc={"zip": 0, "entry": fn}, orig=fn))
                if not missing:
                    stats.media[msg.media[-1].kind] += 1
            if msg.media:
                # strip the attach marker from text, keep an explanatory lead
                msg.text = ATTACH_RE.sub("", body).strip()
            if not msg.system:
                messages.append(msg)
                senders[sender] = senders.get(sender, 0) + 1
                stats.messages += 1
                stats.text_chars += len(msg.text)
            cur = msg
        elif cur and line.strip():
            cur.text = " ".join(x for x in (cur.text, line.strip()) if x)

    if len(senders) > 2:
        stats.group_dropped = 1
        return None, stats

    ranked = sorted(senders.items(), key=lambda kv: -kv[1])
    owner, partner = (ranked[0][0], ranked[1][0]) if len(ranked) > 1 else (ranked[0][0], "")
    thread = Thread(platform="wa", id="", partner=partner, participants=[owner, partner],
                    sources=[f"wa:{p.name}"], messages=messages)
    thread.finalise()
    stats.raw_threads = 1
    return thread, stats