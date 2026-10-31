#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Original export vs standardised bundle — field-by-field diff, per platform.

Two questions, two answers:

  1. side-by-side     what does one message look like on each side (joined on `ts_ms`)
  2. --report         for every field the export carries: does it reach the bundle, in
                      what shape, and how often — plus a message funnel that says *why*
                      a message is absent, not just that it is.

    uv run 26_THESIS/compare_formats.py                        # side-by-side, 2 per class
    uv run 26_THESIS/compare_formats.py --class call --n 5
    uv run 26_THESIS/compare_formats.py --report --out /tmp/diff.md

`ts_ms` is the join key because the bundle strips every name but keeps the time.
Nothing is written into the export or the bundle.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_EXPORT = Path.home() / "Documents/thesis/_active/26_THESIS/data/0_export unzipped/SoMe-chats"
DEFAULT_BUNDLE = Path.home() / "Documents/thesis/_active/26_THESIS/data/1_donation upload-2026-10-27"

WA_LINE = re.compile(r"^[\u200e\u200f\u200b]*\[(\d{2})/(\d{2})/(\d{4}), (\d{2})\.(\d{2})\.(\d{2})\] ([^:]+): ?(.*)$")

# ── originals ───────────────────────────────────────────────────────────────


def load_classic(root: Path, platform: str) -> tuple[list[dict], list[dict]]:
    """Messages and thread headers of an export tree (`messages/<bucket>/<thread>/message_N.json`)."""
    msgs, threads = [], []
    for path in glob.glob(f"{root}/**/message_*.json", recursive=True):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        names = [p.get("name") or "" for p in data.get("participants") or []]
        threads.append({"file": path, "participants": [n for n in names if n], "n": len(data.get("messages") or [])})
        for m in data.get("messages") or []:
            msgs.append({"source": platform, "ts": m.get("timestamp_ms"), "file": path, "raw": m})
    return msgs, threads


def load_e2ee(root: Path) -> tuple[list[dict], list[dict]]:
    msgs, threads = [], []
    for path in glob.glob(f"{root}/Messenger/*.json"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        names = [p for p in data.get("participants") or [] if isinstance(p, str)]
        threads.append({"file": path, "participants": names, "n": len(data.get("messages") or [])})
        for m in data.get("messages") or []:
            msgs.append({"source": "fb-e2ee", "ts": m.get("timestamp"), "file": path, "raw": m})
    return msgs, threads


def load_wa(root: Path) -> tuple[list[dict], list[dict]]:
    import datetime as dt
    msgs, threads = [], []
    for path in sorted(glob.glob(f"{root}/WA/*/*.txt")):
        cur, senders, n = None, Counter(), 0
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            m = WA_LINE.match(line)
            if m:
                dd, mm, yy, hh, mi, ss, sender, body = m.groups()
                ts = int(dt.datetime(int(yy), int(mm), int(dd), int(hh), int(mi), int(ss),
                                     tzinfo=dt.timezone.utc).timestamp() * 1000)
                cur = {"source": "wa", "ts": ts, "file": path, "sender": sender.strip(), "body": body}
                msgs.append(cur)
                senders[sender.strip()] += 1
                n += 1
            elif cur:
                cur["body"] += " " + line.strip()
        threads.append({"file": path, "participants": list(senders), "n": n})
    return msgs, threads


# ── bundle ──────────────────────────────────────────────────────────────────
def load_bundle(root: Path) -> tuple[dict[int, list[dict]], list[dict]]:
    by_ts: dict[int, list[dict]] = defaultdict(list)
    threads = []
    for path in glob.glob(f"{root}/*/*.json"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if "platform" not in data:
            continue
        threads.append({"file": path, "data": data})
        for i, m in enumerate(data.get("messages") or []):
            by_ts[m.get("ts_ms", 0)].append({"thread": data, "path": path, "idx": i, "msg": m})
    return by_ts, threads


# ── the line the model reads is rendered by `src/core/standard/minimal.ts` and nowhere else ──
def stamp(ms: int) -> str:
    if not ms:
        return " " * 20
    import datetime as dt
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


PLACEHOLDER = {"photos": "PHOTO.jpg", "videos": "VIDEO.mp4", "audio": "VOICE.m4a", "other": "FILE"}


def prompt_line(b: dict) -> str:
    m = b["msg"]
    parts = []
    for r in m.get("reactions") or []:
        parts.append(f"<react: {r['emoji']}" + (f" by {r['actor']}" if r.get("actor") else "") + ">")
    counts: dict[str, int] = {}
    for ref in m.get("media") or []:
        name = ref["name"] if ref["kind"] == "audio" else PLACEHOLDER[ref["kind"]]
        counts[name] = counts.get(name, 0) + 1
    parts += [f"<attached: {n} x{c}>" if c > 1 else f"<attached: {n}>" for n, c in counts.items()]
    if "call_s" in m:
        kind = "call" if m["call_media"] == "unknown" else f"{m['call_media']} call"
        parts.append(f"<call: {kind} {m['call_s']}s" + (" missed" if m.get("call_missed") else "") + ">")
    parts += [f"<link: {u}>" for u in m.get("urls") or []]
    if m.get("share"):
        s = m["share"]
        label = " ".join(x for x in (s.get("title"), f"by {s['owner']}" if s.get("owner") else "") if x)
        parts.append(f"<share: {label}>" if label else "<share: post>")
    parts.append("<unsent>") if m.get("unsent") else None
    return f"[{stamp(m['ts_ms'])}] {m['sender']}: {m['text']}" + (f" {' '.join(parts)}" if parts else "")


# ── field flattening ────────────────────────────────────────────────────────
def paths(value, prefix="", depth=0):
    """Every leaf path in a message dict: `reactions[].timestamp`, `sticker.uri`, `photos[].uri`."""
    if depth > 4:
        return
    if isinstance(value, dict):
        for k, v in value.items():
            yield from paths(v, f"{prefix}.{k}" if prefix else k, depth + 1)
    elif isinstance(value, list):
        for v in value:
            yield from paths(v, f"{prefix}[]", depth + 1)
    else:
        yield prefix


def nonempty(v) -> bool:
    return v not in (None, "", [], {}, False, 0)


def repair(s: str) -> str:
    """Meta's double encoding, undone — enough to compare an export string with the bundle's."""
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def same_text(orig: str, got: str) -> bool | None:
    """The bundle's text is the export's text, repaired and with the attach marker cut out."""
    if not nonempty(orig):
        return got == ""
    head = repair(orig).strip()[:40]
    return head in got if head else None


# ── the mapping: what the bundle does with each field ──────────────────────
# status: kept | derived (transformed but present) | policy (deliberately dropped)
#         | lost (should be there, is not) | ignored (constant / duplicate / no information)
#
# `probe` runs against one (original message, bundle message) pair and returns
# True/False, or None when it cannot judge. The report shows the pass rate, so a
# mapping cannot silently rot.
MAPPING: dict[str, dict] = {
    # classic (FB DYI + Instagram)
    "classic": {
        "timestamp_ms": ("kept", "ts_ms", lambda o, b: b["ts_ms"] == o["timestamp_ms"]),
        "sender_name": ("derived", "sender", lambda o, b: b["sender"] in ("SUBJECT", "OTHER-" + b["sender"][-3:])),
        "content": ("kept", "text", lambda o, b: same_text(o.get("content") if isinstance(o.get("content"), str) else "", b["text"])),
        "reactions[].reaction": ("derived", "reactions[].emoji", lambda o, b: len(b.get("reactions") or []) == len(o["reactions"]) and all(r.get("emoji") for r in b["reactions"])),
        "reactions[].actor": ("derived", "reactions[].actor", lambda o, b: len(b.get("reactions") or []) == len(o["reactions"]) and all(r.get("actor") for r in b["reactions"])),
        "reactions[].timestamp": ("kept", "reactions[].ts", lambda o, b: all(r.get("ts") == x.get("timestamp") for r, x in zip(b["reactions"], o["reactions"] if isinstance(o.get("reactions"), list) else []))),
        "reactions[].timestamp": ("kept", "reactions[].ts", lambda o, b: all(r.get("ts") == x.get("timestamp") for r, x in zip(b["reactions"], o["reactions"] if isinstance(o.get("reactions"), list) else []))),
        "photos[].uri": ("derived", "media[].name (kind=photos)", lambda o, b: any(x["kind"] == "photos" for x in b["media"])),
        "videos[].uri": ("derived", "media[].name (kind=videos)", lambda o, b: any(x["kind"] == "videos" for x in b["media"])),
        "audio_files[].uri": ("derived", "media[].name (kind=audio)", lambda o, b: any(x["kind"] == "audio" for x in b["media"])),
        "files[].uri": ("derived", "media[].name (kind=file)", lambda o, b: any(x["kind"] == "file" for x in b["media"])),
        "gifs[].uri": ("derived", "media[].name (kind=gif)", lambda o, b: any(x["kind"] == "gif" for x in b["media"])),
        "sticker.uri": ("derived", "media[].name (kind=sticker)", lambda o, b: any(x["kind"] == "sticker" for x in b["media"])),
        "share.link": ("kept", "share.url + urls[]", lambda o, b: (o["share"].get("link") or "") in (b.get("share") or {}).get("url", "") + "".join(b["urls"])),
        "share.share_text": ("kept", "share.title", lambda o, b: bool((b.get("share") or {}).get("title"))),
        "share.original_content_owner": ("kept", "share.owner", lambda o, b: bool((b.get("share") or {}).get("owner"))),
        "share.profile_share_username": ("kept", "share.owner", lambda o, b: bool((b.get("share") or {}).get("owner"))),
        "share.profile_share_name": ("kept", "share.owner", lambda o, b: bool((b.get("share") or {}).get("owner"))),
        "reactions[].sticker_reaction.uri": ("kept", "reactions[].sticker", lambda o, b: any(r.get("sticker") for r in b.get("reactions") or [])),
        "call_duration": ("kept", "call_s", lambda o, b: b.get("call_s") == o["call_duration"]),
        "missed": ("derived", "call_missed", lambda o, b: b.get("call_missed") is True),
        "is_unsent": ("kept", "unsent", lambda o, b: b.get("unsent") is True),
        "videos[].creation_timestamp": ("ignored", "— (duplicates the message ts)", lambda o, b: None),
        "files[].creation_timestamp": ("ignored", "— (duplicates the message ts)", lambda o, b: None),
        "audio_files[].creation_timestamp": ("ignored", "— (duplicates the message ts)", lambda o, b: None),
        "is_geoblocked_for_viewer": ("ignored", "—", lambda o, b: None),
        "is_unsent_image_by_messenger_kid_parent": ("ignored", "—", lambda o, b: None),
        "ip": ("policy", "— (dropped: IP address)", lambda o, b: None),
        "photos[].creation_timestamp": ("ignored", "— (duplicates the message ts)", lambda o, b: None),
        "sticker.ai_stickers": ("ignored", "—", lambda o, b: None),
    },
    "e2ee": {
        "timestamp": ("kept", "ts_ms", lambda o, b: b["ts_ms"] == o["timestamp"]),
        "senderName": ("derived", "sender", lambda o, b: b["sender"].startswith(("SUBJECT", "OTHER-"))),
        "text": ("kept", "text", lambda o, b: same_text(o.get("text") or "", b["text"])),
        "type": ("ignored", "— (its one non-redundant value, `placeholder`, is `unsent`)", lambda o, b: None),
        "media[].uri": ("derived", "media[].name", lambda o, b: len(b["media"]) == len(o["media"])),
        "reactions[].reaction": ("derived", "reactions[].emoji", lambda o, b: all(r.get("emoji") for r in b.get("reactions") or [])),
        "reactions[].actor": ("derived", "reactions[].actor", lambda o, b: all(r.get("actor") for r in b.get("reactions") or [])),
        "isUnsent": ("kept", "unsent", lambda o, b: b.get("unsent") is True),
        "callDuration": ("kept", "call_s", lambda o, b: b.get("call_s") == o["callDuration"]),
    },
}


def false(o):
    return None


# ── side-by-side ────────────────────────────────────────────────────────────
def classes(orig: list[dict]) -> dict[str, list[dict]]:
    c: dict[str, list[dict]] = defaultdict(list)

    def add(key, m):
        c[key].append(m)

    for m in orig:
        raw, src = m["raw"], m["source"]
        if src == "wa":
            t = m["body"]
            if "end-to-end encrypted" in t:
                add("wa:encryption banner", m)
            if "deleted" in t and len(t) < 40:
                add("wa:deleted message", m)
            if re.search(r"missed (voice|video) call", t, re.I):
                add("wa:missed call", m)
            if re.search(r"(voice|video) call\.", t, re.I):
                add("wa:answered call", m)
            if "<attached:" in t:
                kind = re.search(r"<attached: \d+-([A-Z]+)-", t)
                add(f"wa:attached {kind.group(1) if kind else 'file'}", m)
            if "Location:" in t:
                add("wa:location", m)
            continue
        if raw.get("is_unsent") or raw.get("isUnsent"):
            add("unsent", m)
        if "call_duration" in raw or "missed" in raw:
            add("call", m)
        if raw.get("share"):
            add(f"{src}:share", m)
        for field, key in (("photos", "photo"), ("videos", "video"), ("audio_files", "audio"),
                           ("files", "document"), ("sticker", "sticker"), ("gifs", "gif")):
            if raw.get(field):
                add(f"{src}:{key}", m)
        if raw.get("ip"):
            add(f"{src}:ip", m)
        if raw.get("reactions"):
            add(f"{src}:reaction", m)
    return c


def side_by_side(orig, bundle, only, n) -> str:
    out = ["## Side by side, joined on ts_ms\n"]
    for key in sorted(classes(orig)):
        if only and only not in key:
            continue
        rows = classes(orig)[key]
        with_hit = [m for m in rows if bundle.get(m["ts"])]
        out.append(f"\n### {key} — {len(rows)} in export, {len(with_hit)} reached the bundle\n")
        for m in (with_hit or rows)[:n]:
            hits = bundle.get(m["ts"], [])
            raw = json.dumps(m["raw"], ensure_ascii=False)[:400] if m["source"] != "wa" else json.dumps(m["raw"], ensure_ascii=False)[:220]
            out.append(f"**{m['source']}** `{Path(m['file']).name}`\n")
            out.append(f"- orig: `{raw}`\n")
            if not hits:
                out.append("- bundle: **absent**\n")
                continue
            h = hits[0]
            out.append(f"- bundle `{h['thread']['id']}` `{Path(h['path']).name}`: `{json.dumps(h['msg'], ensure_ascii=False)[:400]}`\n")
        out.append("\n")
    return "".join(out)


# ── full report ─────────────────────────────────────────────────────────────
def funnel(msgs, threads, bundle, floor=10) -> list[str]:
    """Why a message is absent. Every origin at the thread level, then the residual."""
    out = []
    group = floor_drop = kept_threads = 0
    group_msgs = floor_msgs = kept_msgs = 0
    for t in threads:
        senders = {p.strip().lower() for p in t["participants"] if p.strip()}
        if len(senders) > 2:
            group += 1
            group_msgs += t["n"]
        elif t["n"] < floor:
            floor_drop += 1
            floor_msgs += t["n"]
        else:
            kept_threads += 1
            kept_msgs += t["n"]
    out.append(f"| threads | {len(threads)} |")
    out.append("| --- | --- |")
    out.append(f"| dropped: >2 senders | {group} ({group_msgs} msgs) |")
    out.append(f"| dropped: under the {floor}-message floor | {floor_drop} ({floor_msgs} msgs) |")
    out.append(f"| survive the floor | {kept_threads} ({kept_msgs} msgs) |")
    return out


def field_matrix(src: str, msgs: list[dict], bundle: dict[int, list[dict]], mapping: dict) -> tuple[list[str], list[str]]:
    counts: Counter = Counter()
    samples: dict[str, list[dict]] = defaultdict(list)
    for m in msgs:
        hit = bool(bundle.get(m["ts"]))
        for p, v in flatten(m["raw"]):
            if not nonempty(v):
                continue
            counts[p] += 1
            # Probe only where the message survived, or the rate measures the drop, not the mapping.
            if hit and len(samples[p]) < 400:
                samples[p].append(m)
    rows = ["| field | msgs | status | bundle field | probe |", "| --- | --- | --- | --- | --- |"]
    unclassified, lost = [], []
    for p in sorted(counts, key=lambda k: -counts[k]):
        spec = mapping.get(p)
        if not spec:
            rows.append(f"| `{p}` | {counts[p]} | **unclassified** | ? | |")
            unclassified.append(p)
            continue
        status, target, probe = spec
        hits = misses = 0
        for m in samples[p][:120]:
            judged = [probe(m["raw"], h["msg"]) for h in bundle.get(m["ts"], [])[:4]]
            if any(r is True for r in judged):
                hits += 1
            elif any(r is False for r in judged):
                misses += 1
        rate = f"{hits}/{hits + misses}" if hits + misses else "n/a"
        flag = "" if not misses else " **MISS**"
        rows.append(f"| `{p}` | {counts[p]} | {status} | `{target}` | {rate}{flag} |")
        if status == "lost":
            lost.append((p, counts[p], target))
    return rows, unclassified, lost


def flatten(d, prefix=""):
    for p in paths(d):
        cur, ok = d, True
        for part in p.split("."):
            key, _, rest = part.partition("[]")
            try:
                if key and isinstance(cur, dict):
                    cur = cur[key]
                elif part.endswith("[]") and isinstance(cur, list):
                    cur = cur[0] if cur else None
                elif rest and isinstance(cur, list):
                    cur = cur[0] if cur else None
            except Exception:
                ok = False
                break
        yield p, cur if ok else None


def audit(bundle_threads: list[dict]) -> list[str]:
    """Things the field matrix cannot see: empty rows, filename leaks, the audio join."""
    out = ["\n## Audits\n"]
    empty = rows = 0
    same_ts_orphan = 0
    names: set[str] = set()
    audio: dict[str, set[str]] = defaultdict(set)
    prefixed = [0, 0]
    for t in bundle_threads:
        data = t["data"]
        seen: dict[tuple, list[bool]] = defaultdict(list)
        for m in data["messages"]:
            rows += 1
            if not any((m.get("text"), m.get("media"), m.get("reactions"), m.get("share"), m.get("unsent"), m.get("system"))) and "call_s" not in m:
                empty += 1
            seen[(m["ts_ms"], m["sender"])].append(bool(m["text"]))
            for ref in m["media"]:
                names.add(ref["name"])
                if ref["kind"] == "audio":
                    audio[ref["name"]].add(data["id"])
                    if ref["name"].startswith(data["id"] + "-"):
                        prefixed[0] += 1
                    prefixed[1] += 1
        for flags in seen.values():
            if len(flags) > 1 and False in flags and True in flags:
                same_ts_orphan += 1

    leaked = sorted(n for n in names if re.search(r"[A-Za-zÆØÅæøå]{3,}[ ]", n) and not n.lower().endswith(".jpg"))
    shared = [n for n, ids in audio.items() if len(ids) > 1]

    out.append(f"- messages in the bundle: {rows}\n")
    out.append(f"- **empty rows** (no text, media, reaction, call, share, `unsent` or `system`): {empty} — "
               "unknown-media and call-event messages, the only ones left with nothing to say\n")
    out.append(f"- rows sharing `(ts_ms, sender)` with a text-bearing twin (one part of a split export "
               f"holds the text, the other does not): {same_ts_orphan}\n")
    out.append(f"- distinct media names: {len(names)}; audio names in more than one thread: {len(shared)}\n")
    out.append(f"- audio names that begin with their conversation's id: {prefixed[0]}/{prefixed[1]}\n")
    out.append(f"- media names carrying a person's name or a document title ({len(leaked)}) — informational, "
               "the study's legal basis does not require pseudonymisation, and the label map is a courtesy: "
               + ", ".join(f"`{n}`" for n in leaked) + "\n")
    out.append("\n**Audio join.** The stored name begins with the conversation id, so a transcript goes back "
               "to the thread that carried it without scanning the JSONs; inside that file `media[].name` "
               "points at the message and its `ts_ms`.\n")
    return out


def report(orig, threads_by_src, bundle, bundle_threads) -> str:
    out = ["# Export → bundle: field and message accounting\n"]
    total = sum(len(v) for v in threads_by_src.values())
    out.append(f"- original messages: {len(orig)}\n- original threads: {total}\n")
    out.append(f"- bundle messages: {sum(len(v) for v in bundle.values())}\n")
    out.append(f"- bundle threads: {len(bundle_threads)}\n\n")

    out.append("## Message funnel, per source\n")
    for src, threads in threads_by_src.items():
        out.append(f"\n### {src}\n")
        out += [line + "\n" for line in funnel([m for m in orig if m["source"] == src], threads, bundle)]

    all_lost = []
    out.append("\n## Field matrix\n")
    for src, kind in (("fb", "classic"), ("ig", "classic"), ("fb-e2ee", "e2ee")):
        msgs = [m for m in orig if m["source"] == src]
        if not msgs:
            continue
        rows, unclassified, lost = field_matrix(src, msgs, bundle, MAPPING[kind])
        out.append(f"\n### {src} — {len(msgs)} messages\n")
        out += [r + "\n" for r in rows]
        if unclassified:
            out.append(f"\n**unclassified** (needs a decision): {', '.join(f'`{p}`' for p in unclassified)}\n")
        all_lost += [(src, *x) for x in lost]

    out.append("\n### wa — line classes\n")
    wa = [m for m in orig if m["source"] == "wa"]
    k = Counter()
    for m in wa:
        t = m["body"].strip()
        if "end-to-end encrypted" in t:
            k["encryption banner"] += 1
        elif re.search(r"^missed (voice|video) call", t, re.I):
            k["missed call"] += 1
        elif re.search(r"^(voice|video) call\.", t, re.I):
            k["answered call"] += 1
        elif "<attached:" in t:
            ext = re.search(r"<attached: \d+-([A-Z]+)-", t)
            k[f"attached {ext.group(1) if ext else 'file'}"] += 1
        elif "Location:" in t:
            k["location"] += 1
        elif re.search(r"deleted", t, re.I) and len(t) < 40:
            k["deleted notice"] += 1
        else:
            k["text"] += 1
    out.append("| class | msgs | note |\n| --- | --- | --- |\n")
    notes = {
        "encryption banner": "kept as ordinary speech — `system` is not written to the bundle",
        "deleted notice": "`This message was deleted.` does not match SYSTEM_RE; kept as speech",
        "missed call": "recorded only when the line is exactly `Missed voice call`; `… Tap to call back` is not",
        "answered call": "`Voice call. No answer` becomes answered-0s; `1 hr` loses its duration",
        "attached STICKER": "classified by extension → `photos`; prompt says `<attached: PHOTO.jpg>`",
        "attached GIF": "classified by extension → `videos`; prompt says `<attached: VIDEO.mp4>`",
        "attached file": "kind `other`; original filename kept (may carry a person's name)",
    }
    for key, n in k.most_common():
        out.append(f"| {key} | {n} | {notes.get(key, '')} |\n")

    out.append("\n## Lost on the way (field present in the export, absent from the bundle)\n")
    out.append("| source | field | msgs | bundle field |\n| --- | --- | --- | --- |\n")
    for src, field, n, target in all_lost:
        out.append(f"| {src} | `{field}` | {n} | {target} |\n")

    out += audit(bundle_threads)
    out.append("\n## Bundle → source\n")
    out.append("| bundle field | from |\n| --- | --- |\n")
    for row in [
        ("`ts_ms`", "`timestamp_ms` / `timestamp` / the WA line's bracket"),
        ("`sender`", "`sender_name` / `senderName` / WA display name → `SUBJECT`/`OTHER-nnn`"),
        ("`text`", "`content` / `text` / WA body; mojibake repaired, U+200E stripped, attach marker removed"),
        ("`urls[]`", "`extractUrls(text)` ∪ `share.link` — computed at write time, not parse time"),
        ("`reactions[]`", "`reactions[].reaction|actor`; the export's reaction **timestamp is dropped**"),
        ("`share`", "`share.link|share_text|original_content_owner|profile_share_username`"),
        ("`call_s|call_missed|call_media`", "`call_duration|missed` or `Missed voice call` text; absent on most WA variants"),
        ("`media[].name`", "basename of the export uri; audio renamed `NNNNNNNN-AUDIO.ext`"),
        ("`media[].kind`", "export folder (photos/videos/audio/gifs/stickers_used/files) or extension"),
        ("`media[].missing`", "`Failed to download media`, or an `<attached: …>` name not in the zip"),
        ("(never written)", "`is_unsent`/`isUnsent`, `system`, `type`, `ip`, `title`, `thread_path`, reaction timestamps"),
    ]:
        out.append(f"| {row[0]} | {row[1]} |\n")
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    ap.add_argument("--class", dest="only", default=None)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--report", action="store_true", help="field matrix + funnel instead of examples")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fb, fb_t = load_classic(args.export / "FB", "fb")
    ig, ig_t = load_classic(args.export / "IG", "ig")
    e2, e2_t = load_e2ee(args.export)
    wa, wa_t = load_wa(args.export)
    orig = fb + ig + e2 + wa
    bundle, bundle_threads = load_bundle(args.bundle)
    print(f"originals {len(orig)} msgs · bundle {sum(len(v) for v in bundle.values())} msgs", file=sys.stderr)

    if args.report:
        text = report(orig, {"fb": fb_t, "ig": ig_t, "fb-e2ee": e2_t, "wa": wa_t}, bundle, bundle_threads)
    else:
        text = "# Export vs bundle\n" + side_by_side(orig, bundle, args.only, args.n)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
