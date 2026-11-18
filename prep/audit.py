"""Every voice note the prompt has no words for, gathered where a person can listen to them.

A transcript map that covers 857 of 876 voice notes is good enough to stop thinking about — unless the
19 that are left are not silence. This collects them, with the message before and after for context, and
copies the audio out under a name that says when and where it was sent, so the judgement ("this one
really had no words") takes minutes instead of an afternoon of grepping.

    uv run python -B -m prep audit-voice --bundle … --out "…/8_untime-voice"

Unreadable ones are listed too, and marked: a reference with no bytes is a different failure from a
note nobody transcribed. `audio omitted` is the placeholder WhatsApp writes when the attachment never
arrived with the export.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

PLATFORMS = ("fb", "ig", "wa")


def find(bundle: str | Path) -> list[dict]:
    """Voice notes with no transcript, each with a line of surrounding context."""
    root = Path(bundle)
    found: list[dict] = []
    for platform in PLATFORMS:
        directory = root / platform
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            messages = payload.get("messages") or []
            for index, message in enumerate(messages):
                for ref in message.get("media") or []:
                    if ref.get("kind") != "audio" or ref.get("transcript"):
                        continue
                    found.append(
                        {
                            "platform": platform,
                            "partner": payload.get("partner"),
                            "ts_iso": message.get("ts_iso"),
                            "ts_ms": message.get("ts_ms"),
                            "stored": ref.get("name"),
                            "duration_s": ref.get("duration"),
                            "missing_bytes": bool(ref.get("missing")),
                            "before": _line(messages[index - 1]) if index else "",
                            "after": _line(messages[index + 1]) if index + 1 < len(messages) else "",
                        }
                    )
    found.sort(key=lambda r: (r["platform"], r["partner"] or "", r["ts_ms"] or 0))
    return found


def _line(message: dict) -> str:
    return (message.get("as_prompt") or "").replace("\n", " ")[:90]


def gather(bundle: str | Path, out: str | Path) -> dict:
    """Copy the audio to `out/audio/` under readable names, and write `index.md` beside it."""
    root = Path(bundle)
    target = Path(out)
    (target / "audio").mkdir(parents=True, exist_ok=True)
    rows = find(bundle)
    copied = 0
    for row in rows:
        source = root / "media" / (row["stored"] or "")
        usable = bool(row["stored"]) and row["stored"] != "audio omitted" and source.exists()
        row["file"] = ""
        if usable:
            suffix = source.suffix
            name = f"{row['platform']}-{row['partner']}-{row['ts_iso']}-{row['duration_s']}s{suffix}"
            shutil.copy2(source, target / "audio" / name)
            row["file"] = f"audio/{name}"
            copied += 1

    lines = [
        "# Voice notes without a transcript",
        "",
        f"{len(rows)} voice notes carry no words in the prompt; {copied} of them have bytes to listen to.",
        "One line each: where it came from, how long it is, and the messages either side for context.",
        "`no bytes` means the export referenced a file it did not ship — nothing to hear.",
        "",
        "| # | channel · partner | sent | length | file | before → this → after |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for number, row in enumerate(rows, 1):
        length = f"{row['duration_s']}s" if row["duration_s"] is not None else "?"
        file = f"`{row['file']}`" if row["file"] else "**no bytes**"
        context = f"{row['before']} → *voice* → {row['after']}"
        lines.append(f"| {number} | {row['platform']} · {row['partner']} | {row['ts_iso']} | {length} | {file} | {context} |")
    (target / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"voice_notes": len(rows), "with_bytes": copied, "no_bytes": len(rows) - copied, "out": str(out)}
