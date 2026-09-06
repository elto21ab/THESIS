"""Facebook Messenger E2EE "Message storage" dump reader.

Flat set: one `<Partner Name>_<id>.json` per 1:1 chat + `media/` folder.
Schema differs from DYI:
  participants: string[]            (DYI uses [{name}])
  message: { senderName, timestamp, text, type, media:[{uri}], reactions, isUnsent }
Timestamps are epoch ms — same family as DYI `timestamp_ms`, so cross-source dedupe
by ts is valid. Media uris are `./media/<uuid>.<ext>`; `"Failed to download media"`
messages are flagged missing, never silently dropped.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from format import MediaRef, Msg, Reaction, SourceStats, Thread, media_kind_of, repair_mojibake

DOWNLOAD_FAILED = "Failed to download media"


def read_e2ee(input_path: str | Path) -> tuple[list[Thread], SourceStats]:
    """input_path: the messages.zip OR the unzipped messages/ folder."""
    stats = SourceStats("fb")
    files: list[tuple[str, bytes]] = []

    p = Path(input_path)
    if p.is_dir():
        stats.archives = 1
        files = [(f.name, f.read_bytes()) for f in sorted(p.glob("*.json"))]
    elif p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as z:
            stats.archives = 1
            files = [(n.split("/")[-1], z.read(n)) for n in z.namelist() if n.endswith(".json") and not n.endswith("/")]
    elif p.suffix.lower() == ".json":
        stats.archives = 1
        files = [(p.name, p.read_bytes())]
    else:
        raise ValueError(f"e2ee input must be a folder of json, a zip, or a json: {p}")

    threads: list[Thread] = []
    for fname, raw in files:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            stats.group_dropped += 1  # unparseable counts as not-a-thread
            continue
        participants = [repair_mojibake(p.strip())[0] for p in (data.get("participants") or []) if p and p.strip()]
        if len(participants) > 2:
            stats.group_dropped += 1
            continue
        if not participants:
            continue

        msgs: list[Msg] = []
        for m in data.get("messages") or []:
            text, repaired = repair_mojibake((m.get("text") or "").replace("\u200e", ""))
            stats.escapes_repaired += repaired
            media: list[MediaRef] = []
            for md in m.get("media") or []:
                uri = md.get("uri", "")
                if uri == DOWNLOAD_FAILED:
                    media.append(MediaRef(uri="", kind="other", missing=True))
                elif uri:
                    rel = uri.removeprefix("./")
                    media.append(MediaRef(uri=rel, kind=media_kind_of(uri),
                                          loc={"file": str(p / rel)} if p.is_dir() else None,
                                          orig=rel))
            for x in media:
                if not x.missing:
                    stats.media[x.kind] += 1
            if m.get("isUnsent"):
                continue
            msgs.append(Msg(ts=int(m.get("timestamp") or 0), sender=(m.get("senderName") or "").strip(),
                            text=text, media=media, reactions=[Reaction(emoji=r.get("reaction") or "", actor=r.get("actor") or "") for r in (m.get("reactions") or [])]))
            stats.messages += 1
            stats.text_chars += len(text)

        t = Thread(platform="fb", id="", partner=participants[0] if participants else "",
                   participants=participants, sources=["fb-e2ee-dump"], messages=msgs)
        t.finalise()
        stats.raw_threads += 1
        threads.append(t)
    return threads, stats