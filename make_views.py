#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Two inspection views of one corpus, side by side with the two things that already exist.

    0_export raw/        the export as it came out of Facebook / Instagram / WhatsApp  (untouched)
    1_export readable/   ← this script: one file per conversation, all export parts merged
    2_donation/          what the tool produced (de-identified, filtered)              (untouched)
    3_rendered/          ← this script: the exact text the model will read

Nothing is moved and nothing is overwritten: the two new folders are written fresh, and the
existing ones are only read.

    uv run 26_THESIS/make_views.py \
        --export "26_THESIS/data/0_export raw" \
        --bundle "26_THESIS/data/2_donation" \
        --out    "26_THESIS/data"
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

MESSAGE_RE = re.compile(r"messages/(?P<bucket>[^/]+)/(?P<thread>[^/]+)/message_\d+\.json$")
WA_LINE = re.compile(r"^[\u200e\u200f\u200b]*\[([^\]]+)\] ([^:]+): ?(.*)$")
ATTACH_RE = re.compile(r"<attached:\s*([^>]+)>")


def iso(ms: int | None) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def safe(name: str) -> str:
    """A file name that keeps the name readable: diacritics and spaces survive, slashes do not."""
    return re.sub(r"[/:]", "-", name).strip() or "unnamed"


# ── 1_export readable ────────────────────────────────────────────────────────
class Conversation:
    def __init__(self, platform: str, key: str) -> None:
        self.platform = platform
        self.key = key
        self.names: set[str] = set()
        self.rows: list[dict] = []
        self.sources: set[str] = set()

def read_meta(root: Path, platform: str, prefix: str) -> dict[str, Conversation]:
    taken: dict[str, Conversation] = {}
    files = sorted(glob.glob(f"{root}/{prefix}/**/message_*.json", recursive=True))
    for path in files:
        rel = Path(path).as_posix()
        match = MESSAGE_RE.search(rel)
        if not match:
            continue
        key = f"{match.group('bucket')}/{match.group('thread')}"
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        conv = taken.setdefault(key, Conversation(platform, key))
        conv.sources.add(rel.split("your_")[0] or "part")
        for participant in data.get("participants") or []:
            name = (participant or {}).get("name") if isinstance(participant, dict) else participant
            if name:
                conv.names.add(name.strip())
        for m in data.get("messages") or []:
            row = dict(m)
            row["_who"] = (m.get("sender_name") or m.get("senderName") or "").strip()
            row["_ts"] = m.get("timestamp_ms") or m.get("timestamp") or 0
            row["_src"] = rel
            conv.rows.append(row)
    return taken


def read_e2ee(root: Path) -> dict[str, Conversation]:
    taken: dict[str, Conversation] = {}
    for path in sorted(glob.glob(f"{root}/Messenger/*.json")):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            continue
        key = Path(path).stem
        conv = Conversation("e2ee", key)
        conv.sources.add(Path(path).name)
        for name in data.get("participants") or []:
            if isinstance(name, str) and name.strip():
                conv.names.add(name.strip())
        for m in data.get("messages") or []:
            row = dict(m)
            row["_who"] = (m.get("senderName") or "").strip()
            row["_ts"] = m.get("timestamp") or 0
            row["_src"] = Path(path).name
            conv.rows.append(row)
        taken[key] = conv
    return taken


def write_readable(export: Path, out: Path) -> list[str]:
    log: list[str] = []
    groups = {
        "fb": read_meta(export, "fb", "FB"),
        "ig": read_meta(export, "ig", "IG"),
        "e2ee": read_e2ee(export),
    }
    for platform, convs in groups.items():
        target = out / "1_export readable" / platform
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for key, conv in sorted(convs.items()):
            conv.rows.sort(key=lambda r: r["_ts"])
            names = sorted(conv.names)
            partner = next((n for n in names if n not in ("Elias Salvador Smidt Torjani",)), names[0] if names else key)
            path = target / f"{safe(partner)}.jsonl"
            if path.exists():
                path = target / f"{safe(partner)}__{safe(key.split('/')[-1])}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for row in conv.rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            ts = [r["_ts"] for r in conv.rows if r["_ts"]]
            rows.append((partner, len(conv.rows), iso(min(ts)) if ts else "", iso(max(ts)) if ts else "",
                         len({r["_src"] for r in conv.rows}), path.name))
        rows.sort(key=lambda r: -r[1])
        with (target / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {platform} — {len(rows)} conversations, {sum(r[1] for r in rows):,} messages\n\n")
            fh.write("One file per conversation, every export part merged and sorted by time. "
                     "`parts` = how many files of the export contributed.\n\n")
            fh.write("| conversation | messages | first | last | parts | file |\n| --- | --- | --- | --- | --- | --- |\n")
            for row in rows:
                fh.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | `{row[5]}` |\n")
        log.append(f"{platform}: {len(rows)} conversations, {sum(r[1] for r in rows):,} messages")

    wa_src = export / "WA"
    if wa_src.is_dir():
        target = out / "1_export readable" / "wa"
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for chat in sorted(p for p in wa_src.iterdir() if p.is_dir()):
            txt = next(iter(chat.glob("*.txt")), None)
            if not txt:
                continue
            dest = target / safe(chat.name)
            dest.mkdir(exist_ok=True)
            shutil.copy2(txt, dest / "_chat.txt")
            attachments = sorted(p.name for p in chat.iterdir() if p.is_file() and p.suffix != ".txt")
            (dest / "attachments.txt").write_text("\n".join(attachments) + "\n", encoding="utf-8")
            lines = [ln for ln in txt.read_text(encoding="utf-8", errors="replace").splitlines() if WA_LINE.match(ln)]
            rows.append((chat.name, len(lines), len(attachments)))
        rows.sort(key=lambda r: -r[1])
        with (target / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# wa — {len(rows)} chats, {sum(r[1] for r in rows):,} timestamped lines\n\n")
            fh.write("WhatsApp already exports one chat per folder as text, so the readable view is that text "
                     "beside a list of the attachments it names.\n\n")
            fh.write("| chat | lines | attachments |\n| --- | --- | --- |\n")
            for row in rows:
                fh.write(f"| {row[0]} | {row[1]} | {row[2]} |\n")
        log.append(f"wa: {len(rows)} chats")
    return log


# ── 3_rendered ───────────────────────────────────────────────────────────────
def write_rendered(bundle: Path, out: Path) -> list[str]:
    target = out / "3_rendered"
    if target.exists():
        shutil.rmtree(target)
    log: list[str] = []
    for platform in ("fb", "ig", "wa"):
        src = bundle / platform
        if not src.is_dir():
            continue
        dest = target / platform
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        for path in sorted(src.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = f"# {data['platform']} {data['partner']}"
            lines = []
            legacy = 0
            for m in data.get("messages") or []:
                if "as_prompt" in m:
                    lines.append(m["as_prompt"])
                    continue
                # A bundle built before the rendered text was carried has none of it. Say the plainest
                # true thing rather than re-implementing the renderer here: a second renderer is what
                # this whole exercise removed.
                legacy += 1
                stamp = iso(m.get("ts_ms"))
                lines.append(f"[{stamp}] {m.get('sender', '')}: {m.get('text', '')}".rstrip())
            if legacy:
                log.append(f"{platform}/{path.stem}: {legacy} messages have no `as_prompt` — rebuild the bundle to see the rendered text")
            text = "\n".join([name] + lines)
            (dest / path.name.replace(".json", ".txt")).write_text(text + "\n", encoding="utf-8")
            rows.append((data["partner"], len(data.get("messages") or []), data.get("first", ""), data.get("last", "")))
        rows.sort(key=lambda r: -r[1])
        with (dest / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {platform} — {len(rows)} conversations, {sum(r[1] for r in rows):,} messages\n\n")
            fh.write("`<partner>.txt` is what the model reads for that conversation: the chunk header and "
                     "one block per message. A block may span several lines — its text keeps its own "
                     "newlines — so split on the `[` at the start of a line, never on `\\n` alone.\n\n")
            fh.write("| partner | messages | first | last |\n| --- | --- | --- | --- |\n")
            for row in rows:
                fh.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n")
        log.append(f"{platform}: {len(rows)} conversations rendered")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    lines = write_readable(args.export, args.out) + write_rendered(args.bundle, args.out)
    for line in lines:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
