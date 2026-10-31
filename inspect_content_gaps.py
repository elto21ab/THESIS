#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Two questions about what the export didn't say, answered from a bundle plus the export it came from.

  1. which conversations have a partner Meta refuses to name (`partner_unnamed`), and what is in them
  2. messages that arrive as `<no content>`, in the context of the four around them

The bundle has labels instead of names, so every example is joined back to the original export by
timestamp — that is the only key the two sides share — and the export's own file path and sender
name are printed beside it. Nothing is written anywhere.

    uv run 26_THESIS/inspect_content_gaps.py --bundle /tmp/b15 \\
        --export "26_THESIS/data/0_export unzipped/SoMe-chats"
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_EXPORT = Path.home() / "Documents/thesis/_active/26_THESIS/data/0_export unzipped/SoMe-chats"
CONTEXT = 3


def export_index(root: Path) -> dict[int, tuple[str, str]]:
    """ts → (file, sender as the export spells it), for joining a bundle row back to its source."""
    index: dict[int, tuple[str, str]] = {}
    for pattern in ("FB/**/message_*.json", "IG/**/message_*.json", "Messenger/*.json"):
        for path in glob.glob(f"{root}/{pattern}", recursive=True):
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            for m in data.get("messages") or []:
                ts = m.get("timestamp_ms") or m.get("timestamp")
                if ts is not None:
                    index[int(ts)] = (path.replace(str(root) + "/", ""), m.get("sender_name") or m.get("senderName") or "")
    return index


def threads(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.glob("*/*.json")):
        if path.name == "report.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "platform" in data:
            data["_path"] = str(path)
            out.append(data)
    return out


def where(index: dict[int, tuple[str, str]], ts: int) -> str:
    hit = index.get(ts)
    return f"{hit[0]}  as {hit[1]!r}" if hit else "(not in the export)"


def unnamed(bundle: list[dict], index: dict[int, tuple[str, str]], per: int) -> list[str]:
    out = []
    rows = [t for t in bundle if t.get("partner_unnamed")]
    out.append(f"## Conversations with an unnamed partner — {len(rows)}\n")
    for t in rows:
        msgs = t["messages"]
        empty = sum(1 for m in msgs if "<no content>" in m["as_prompt"])
        out.append(f"\n### `{t['platform']}/{t['partner']}` · {len(msgs)} messages · {empty} empty"
                   f" · {t['first']} → {t['last']} · {','.join(t['sources'])}\n")
        out.append(f"- original: `{where(index, msgs[0]['ts_ms'])}`\n")
        if len(msgs) > 1:
            out.append(f"- another row: `{where(index, msgs[len(msgs) // 2]['ts_ms'])}`\n")
        out.append("\n")
        for m in msgs[:per]:
            out.append(f"    {m['as_prompt'][:150]}\n")
    return out


def empties(bundle: list[dict], index: dict[int, tuple[str, str]], want: int) -> list[str]:
    out = [f"\n## `<no content>` in context — first {want} of "
           f"{sum(1 for t in bundle for m in t['messages'] if '<no content>' in m['as_prompt'])}\n"]
    shown = 0
    for t in bundle:
        msgs = t["messages"]
        for i, m in enumerate(msgs):
            if "<no content>" not in m["as_prompt"]:
                continue
            gap = (m["ts_ms"] - msgs[i - 1]["ts_ms"]) / 1000 if i else 0
            who = "the donor" if m["sender"] == "SUBJECT" else "the partner"
            unnamed_thread = " · unnamed partner" if t.get("partner_unnamed") else ""
            out.append(f"\n### `{t['platform']}/{t['partner']}`{unnamed_thread} · empty message from {who}"
                       f" · {m['ts_iso']} · {gap:.0f}s after the one above\n")
            out.append(f"- original: `{where(index, m['ts_ms'])}`\n\n")
            for k in range(max(0, i - CONTEXT), min(len(msgs), i + CONTEXT + 1)):
                mark = "→" if k == i else " "
                out.append(f"  {mark} {msgs[k]['as_prompt'][:150]}\n")
            shown += 1
            if shown >= want:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--per-thread", type=int, default=4)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    index = export_index(args.export)
    bundle = threads(args.bundle)
    print(f"{len(bundle)} conversations, {len(index)} export messages indexed", file=sys.stderr)
    text = "".join(unnamed(bundle, index, args.per_thread) + empties(bundle, index, args.examples))
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
