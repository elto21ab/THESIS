"""Classic Meta export reader — the messages/<bucket>/<thread>/message_N.json tree
shared by Facebook DYI and Instagram.

Same schema on both: participants [{name}], messages with
sender_name / timestamp_ms / content / photos[] / videos[] / audio_files[].
FB splits histories across zips at 2 GiB wherever the byte budget runs out; threads are
grouped by folder path across every archive. Input may be a set of zips, or an unzipped
directory tree (Google Drive sync folders, Instagram activity folder).
Disk walk prunes known media dirs so Google Drive sync folders are not fully statted.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Iterator

from format import MediaRef, Msg, Platform, Reaction, SourceStats, Thread, media_kind_of, repair_mojibake

MESSAGE_RE = re.compile(r"messages/(?P<bucket>[^/]+)/(?P<thread>[^/]+)/message_\d+\.json$")
PRUNE = {"photos", "videos", "audio", "gifs", "stickers", "media", "files"}


def _iter_disk_messages(root: Path) -> Iterator[tuple[str, str, bytes, Path]]:
    """Yield (norm_path, bucket, bytes, real_file)."""
    for dirpath, dirnames, filenames in root.walk():
        dirnames[:] = [d for d in dirnames if d not in PRUNE]
        for f in filenames:
            if not f.startswith("message_"):
                continue
            full = Path(dirpath) / f
            rel = full.relative_to(root).as_posix()
            # Normalise to the classic pattern. If the root IS the messages/ tree
            # itself, rel already starts with a bucket dir — prepend messages/.
            idx = rel.find("messages/")
            norm = rel[idx:] if idx != -1 else f"messages/{rel}"
            m = MESSAGE_RE.match(norm)
            if m:
                yield norm, m.group("bucket"), full.read_bytes(), full


def read_classic(source: str | Path, platform: Platform) -> tuple[list[Thread], SourceStats]:
    """source: zip file, a directory tree containing messages/…, or a single message JSON."""
    stats = SourceStats(platform)
    by_dir: dict[str, dict] = {}
    src = Path(source)
    src_root = src if src.is_dir() else src.parent

    def ingest(entry_path: str, raw: bytes, zip_idx: int | None = None, zip_name: str = "",
               real_file: Path | None = None) -> None:
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        dk = entry_path.rsplit("/", 1)[0]  # …/messages/<bucket>/<thread>
        b = by_dir.setdefault(dk, {"names": set(), "thread_dir": dk, "messages": [],
                                   "zip_idx": zip_idx, "zip_name": zip_name})
        if real_file is not None:
            b["media_root"] = real_file.parent
        for pn in data.get("participants") or []:
            if pn.get("name"):
                b["names"].add(repair_mojibake(pn["name"].strip())[0])
        for m in data.get("messages") or []:
            text = m.get("content")
            if isinstance(text, list):
                text = " ".join(c for c in text if isinstance(c, str))
            text = text or ""
            text, repaired = repair_mojibake(text.replace("\u200e", ""))
            stats.escapes_repaired += repaired
            media: list[MediaRef] = []
            for grp, kind in (("photos", "photos"), ("videos", "videos"), ("audio_files", "audio")):
                for it in m.get(grp) or []:
                    uri = it.get("uri") or ""
                    if uri:
                        relu = uri.removeprefix("./")
                        if b["zip_idx"] is not None:
                            loc = {"zip": b["zip_idx"], "entry": dk + "/" + relu}
                        elif b.get("media_root"):
                            # IG/disk uris embed the activity root; media sits in
                            # thread_dir/<photos|videos|audio>/<name>.
                            tail = re.search(r"/(photos|videos|audio)/[^/]+$", relu)
                            loc = {"file": str(b["media_root"] / (tail.group(0).lstrip("/") if tail else Path(relu).name))}
                        else:
                            loc = None
                        media.append(MediaRef(uri=relu, kind=kind, loc=loc, orig=relu))
            for x in media:
                if x.loc:
                    stats.media[x.kind] += 1
            reactions = []
            for r in m.get("reactions") or []:
                reactions.append(Reaction(emoji=r.get("reaction") or "", actor=r.get("actor") or "", ts=int(r.get("timestamp") or 0)))
            b["messages"].append(
                Msg(ts=int(m.get("timestamp_ms") or 0), sender=(m.get("sender_name") or "").strip(),
                    text=text, media=media, reactions=reactions)
            )
            stats.messages += 1
            stats.text_chars += len(text)

    src_root = src
    if src.is_dir():
        zips = sorted(src.glob("*.zip"))
        if zips:
            # A folder of split-export zips (FB DYI) — read every part.
            stats.archives = len(zips)
            for zi, zf in enumerate(zips):
                with zipfile.ZipFile(zf) as z:
                    for n in z.namelist():
                        if n.endswith("/"):
                            continue
                        m = MESSAGE_RE.search(n)
                        if m:
                            ingest(n, z.read(n), zip_idx=zi, zip_name=zf.name)
        else:
            # plain folder of message jsons OR export tree
            stats.archives = 1
            for entry, bucket, raw, real in _iter_disk_messages(src):
                ingest(entry, raw, real_file=real)
    elif src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            stats.archives = 1
            for n in z.namelist():
                if n.endswith("/"):
                    continue
                m = MESSAGE_RE.search(n)
                if m:
                    ingest(n, z.read(n), zip_idx=0, zip_name=src.name)
    elif src.suffix.lower() == ".json":
        # single message_N.json
        stats.archives = 1
        entry = src.name
        ingest(entry, src.read_bytes())
    else:
        raise ValueError(f"unsupported input: {src}")

    threads: list[Thread] = []
    for dk, b in by_dir.items():
        if len(b["names"]) > 2:
            stats.group_dropped += 1
            continue
        if not b["names"]:
            continue
        parts = dk.split("/")
        bucket = parts[parts.index("messages") + 1] if "messages" in parts else "inbox"
        t = Thread(platform=platform, id="", partner="",
                   participants=sorted(b["names"]), sources=[f"{platform}-{bucket}"],
                   messages=b["messages"],
                   media_zip=b.get("zip_idx"), media_base=dk)
        t.finalise()
        stats.raw_threads += 1
        threads.append(t)
    return threads, stats