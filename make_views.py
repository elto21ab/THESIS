#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Rebuild the two inspection views of one corpus from one bundle.

    0_export unzipped/   the export as it came out of the platforms          (untouched)
    1_export readable/   ← this: the conversations the bundle kept, in the export's own format,
                             named by the bundle's own file name, so the pair lines up 1:1
    2_donation …/        what the tool produced                              (untouched)
    3_rendered/          ← this: the exact text the model reads
    4_chunks/            ← this: the logical-day chunks, ids and token counts, no embeddings

The readable view is *derived from the bundle's decisions*: same conversations (groups dropped, floor
applied, Facebook's two exports merged), same order, same file names. What it does not do is mask:
inside each file the messages are the export's own JSON, real names and original keys.

    uv run 26_THESIS/make_views.py \\
        --export "26_THESIS/data/0_export unzipped/SoMe-chats" \\
        --bundle "26_THESIS/data/2_donation upload-2026-10-27-v3" \\
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


def iso(ms: int | None) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


# ── the export, indexed by timestamp ─────────────────────────────────────────
def export_index(export: Path) -> tuple[dict[int, list[dict]], dict[int, str]]:
    """ts → the export's own message objects, and ts → which part it came from.

    A list per timestamp, because two messages can share a millisecond; the bundle's order decides
    which is which when that happens.
    """
    rows: dict[int, list[dict]] = defaultdict(list)
    source: dict[int, str] = {}
    patterns = ("FB/**/message_*.json", "IG/**/message_*.json", "Messenger/*.json")
    for pattern in patterns:
        for path in glob.glob(f"{export}/{pattern}", recursive=True):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            rel = Path(path).as_posix().replace(f"{export.as_posix()}/", "")
            for m in data.get("messages") or []:
                ts = m.get("timestamp_ms") or m.get("timestamp")
                if ts is None:
                    continue
                rows[int(ts)].append(m)
                source.setdefault(int(ts), rel)
    return rows, source


# ── 1_export readable ────────────────────────────────────────────────────────
def write_readable(bundle: Path, rows: dict[int, list[dict]], source: dict[int, str], out: Path) -> list[str]:
    target = out / "1_export readable"
    if target.exists():
        shutil.rmtree(target)
    log: list[str] = []
    for platform in ("fb", "ig"):
        src = bundle / platform
        if not src.is_dir():
            continue
        dest = target / platform
        dest.mkdir(parents=True, exist_ok=True)
        table = []
        for path in sorted(src.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            lines, missing = [], 0
            for m in data.get("messages") or []:
                bucket = rows.get(m["ts_ms"]) or []
                # Take the export object this bundle row came from; pop so a shared millisecond is not
                # handed to two rows. Order is chronological on both sides, so first-in is the match.
                picked = bucket.pop(0) if bucket else None
                if picked is None:
                    missing += 1
                    lines.append({"_ts_ms": m["ts_ms"], "_missing_in_export": True, "_bundle": m})
                    continue
                lines.append({**picked, "_src": source.get(m["ts_ms"], "")})
            # The export's own envelope: one object per conversation, `participants` and `messages`,
            # each message the export's own JSON with `_src` added. Not one object per line — the
            # point of this folder is that it is the export, only merged.
            # The export's own participant names, taken from the messages: the bundle's list is
            # labels, and this folder exists precisely to show the names behind them.
            names = sorted({
                (line.get("sender_name") or line.get("senderName") or "").strip()
                for line in lines
                if (line.get("sender_name") or line.get("senderName") or "").strip()
            })
            envelope = {
                "participants": names,
                "_note": "the export's own messages, all parts merged and sorted; _src names the part",
                "messages": lines,
            }
            with (dest / path.name).open("w", encoding="utf-8") as fh:
                json.dump(envelope, fh, ensure_ascii=False, indent=1)
            table.append((path.stem, len(lines), data.get("first", ""), data.get("last", ""), missing))
        with (dest / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {platform} — {len(table)} conversations\n\n")
            fh.write("Same conversations, same order and **same file names** as the bundle in "
                     "`2_donation/`, but with the export's own fields and real names. "
                     "`_src` names the export part the message came from.\n\n")
            fh.write("| file | messages | first | last | not found in export |\n| --- | --- | --- | --- | --- |\n")
            for row in table:
                fh.write(f"| `{row[0]}.json` | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n")
        log.append(f"{platform}: {len(table)} files")
    log += write_readable_wa(bundle, out)
    return log


WA_STAMP = re.compile(r"^[\u200e\u200f\u200b]*\[(\d{2})/(\d{2})/(\d{4}), (\d{2})\.(\d{2})\.(\d{2})\]")


def write_readable_wa(bundle: Path, out: Path) -> list[str]:
    """WhatsApp already exports one chat per folder as text, so this view stays text.

    The folder is named by the bundle's label — the archive's own name is `WhatsApp Chat - <person>`
    and is not in the bundle — and the label is resolved by matching timestamps against the bundle's
    thread, because that is the only key the two sides share.
    """
    import datetime as dt

    src = bundle / "wa"
    if not src.is_dir():
        return []
    stamps: dict[int, str] = {}
    for path in sorted(src.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for m in data.get("messages") or []:
            stamps[m["ts_ms"]] = data["partner"]
    target = out / "1_export readable" / "wa"
    target.mkdir(parents=True, exist_ok=True)
    chats = sorted(p for p in (out / "0_export unzipped" / "SoMe-chats" / "WA").iterdir() if p.is_dir())
    table = []
    for chat in chats if chats else []:
        txt = next(iter(chat.glob("*.txt")), None)
        if not txt:
            continue
        votes: dict[str, int] = defaultdict(int)
        ours = 0
        for line in txt.read_text(encoding="utf-8", errors="replace").splitlines():
            hit = WA_STAMP.match(line)
            if not hit:
                continue
            dd, mm, yy, hh, mi, ss = (int(g) for g in hit.groups())
            ts = int(dt.datetime(yy, mm, dd, hh, mi, ss, tzinfo=dt.timezone.utc).timestamp() * 1000)
            label = stamps.get(ts)
            if label:
                votes[label] += 1
            else:
                ours += 1
        name = max(votes, key=votes.get) if votes else chat.name
        dest = target / name
        dest.mkdir(exist_ok=True)
        shutil.copy2(txt, dest / "_chat.txt")
        attachments = sorted(p.name for p in chat.iterdir() if p.is_file() and p.suffix != ".txt")
        (dest / "attachments.txt").write_text("\n".join(attachments) + "\n", encoding="utf-8")
        table.append((name, chat.name, len(attachments), max(votes.values()) if votes else 0))
    with (target / "index.md").open("w", encoding="utf-8") as fh:
        fh.write(f"# wa — {len(table)} chats\n\nFolder name = the bundle's label; the archive's own "
                 "name is beside it. `matched` is how many of its lines line up with the bundle's "
                 "thread, which is how the label was found.\n\n")
        fh.write("| label | export folder | attachments | matched lines |\n| --- | --- | --- | --- |\n")
        for row in table:
            fh.write(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} |\n")
    return [f"wa: {len(table)} chats"]


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
        table = []
        for path in sorted(src.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            messages = data.get("messages") or []
            legacy = sum(1 for m in messages if "as_prompt" not in m)
            lines = [f"# {data['platform']} {data['partner']}"]
            for m in messages:
                if "as_prompt" in m:
                    lines.append(m["as_prompt"])
                else:
                    lines.append(f"[{iso(m.get('ts_ms'))}] {m.get('sender', '')}: {m.get('text', '')}".rstrip())
            (dest / path.name.replace(".json", ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
            table.append((path.stem, len(messages), legacy))
        with (dest / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {platform} — {len(table)} conversations\n\n")
            fh.write("Exactly what the model reads, one block per message. A block may span several "
                     "lines — its text keeps its own newlines — so never split on `\\n` alone.\n\n")
            fh.write("| file | messages | rendered by the old format |\n| --- | --- | --- |\n")
            for row in table:
                fh.write(f"| `{row[0]}.txt` | {row[1]} | {row[2]} |\n")
        log.append(f"{platform}: {len(table)} rendered")
    return log


# ── 4_chunks ─────────────────────────────────────────────────────────────────
def write_chunks(bundle: Path, out: Path, chunker: str = "day") -> list[str]:
    """The chunks the study would embed: logical-day or fixed windows, no embeddings made."""
    sys.path.insert(0, str(Path(__file__).parent / "retrieval"))
    from retrieval.bundle import load_bundle  # noqa: PLC0415
    from retrieval.chunk import CHUNKERS, ChunkCfg, RENDER_VERSION  # noqa: PLC0415

    threads, rejects, report = load_bundle(bundle)
    count = lambda t: max(1, len(t) // 4)  # noqa: E731 — chars/4, the same proxy the CLI uses
    target = out / "4_chunks"
    if target.exists():
        shutil.rmtree(target)
    log = [f"{chunker} chunker · render {RENDER_VERSION} · {len(threads)} conversations, {len(rejects)} rejected by the loader"]
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for platform in ("fb", "ig", "wa"):
        dest = target / platform
        dest.mkdir(parents=True, exist_ok=True)
        rows = []
        for thread in (t for t in threads if t.platform == platform):
            chunks = CHUNKERS[chunker](thread, ChunkCfg(), count)
            body = [f"# {thread.platform} {thread.partner} — {len(chunks)} chunks, "
                    f"{len(thread.messages)} messages"]
            for c in chunks:
                body.append(f"\n<!-- {c.id} · seq {c.seq} · msgs {c.i0}..{c.i1 - 1} · {c.n_msgs} · "
                            f"~{c.n_tokens} tok · {c.day} -->")
                body.append(c.text)
            (dest / f"{thread.partner}.md").write_text("\n".join(body) + "\n", encoding="utf-8")
            rows.append((thread.partner, len(chunks), sum(c.n_tokens for c in chunks)))
            totals[platform][0] += 1
            totals[platform][1] += len(chunks)
            totals[platform][2] += sum(c.n_tokens for c in chunks)
        rows.sort(key=lambda r: -r[1])
        with (dest / "index.md").open("w", encoding="utf-8") as fh:
            fh.write(f"# {platform} — {len(rows)} conversations, {totals[platform][1]} chunks, "
                     f"~{totals[platform][2]:,} tokens\n\n")
            fh.write("`<partner>.md` holds that conversation's chunks in order. The comment above each "
                     "chunk carries its id (keyed on the render version), its message span, and its "
                     "size; a chunk with nothing but `<no content>` is not built at all.\n\n")
            fh.write("| conversation | chunks | ~tokens |\n| --- | --- | --- |\n")
            for row in rows:
                fh.write(f"| `{row[0]}` | {row[1]} | {row[2]:,} |\n")
    (target / "index.md").write_text(
        "# Chunks\n\n" + "\n".join(f"- {line}" for line in log)
        + "\n\nNothing is embedded here: the text, the span and the token count are what the index "
          "would be built from. `retrieval` owns this step; this folder is a copy to read.\n",
        encoding="utf-8")
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-readable", action="store_true")
    ap.add_argument("--skip-rendered", action="store_true")
    ap.add_argument("--chunks", default="", choices=["", "day", "fixed"], help="also write 4_chunks")
    args = ap.parse_args()
    log = [] if args.skip_readable else write_readable(args.bundle, *export_index(args.export), args.out)
    log += [] if args.skip_rendered else write_rendered(args.bundle, args.out)
    if args.chunks:
        log += write_chunks(args.bundle, args.out, args.chunks)
    for line in log:
        print(line, file=sys.stderr)


if __name__ == "__main__":
    main()
