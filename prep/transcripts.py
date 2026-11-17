"""Attach speech-to-text output to a bundle, where the prompt will read it.

A voice note renders as `<voice 0:12>` until somebody transcribes it and `<voice>the words</voice>`
after. Transcription is slow and the data pipeline is not, so the two were deliberately kept apart: an
STT pass writes a `{stored_name: text}` map, and this puts each line where the renderer already looks —
`media[].transcript`, matched on `media[].name`.

That name is the join. Nothing else needs to be stable, and nothing else is used:

    bundle message          media[0].name = "fb001-00000004-AUDIO.mp4"
    transcripts.json        {"fb001-00000004-AUDIO.mp4": "Hej med dig"}

A transcript for a file the bundle does not hold is reported, not guessed at; a voice note with no
transcript keeps its duration. Re-rendering is left to `prep render`, so this stage only writes the
field and the line follows — one decision per verb, and a diff that says what changed.
"""
from __future__ import annotations

import json
from pathlib import Path

PLATFORMS = ("fb", "ig", "wa")


def attach(payload: dict, transcripts: dict[str, str], used: set[str]) -> int:
    """Set `transcript` on every audio reference whose stored name has one. Returns how many."""
    attached = 0
    for message in payload.get("messages") or []:
        for ref in message.get("media") or []:
            if ref.get("kind") != "audio":
                continue
            text = transcripts.get(ref.get("name") or "")
            if not text:
                continue
            ref["transcript"] = text
            used.add(ref["name"])
            attached += 1
    return attached


def apply(bundle: str | Path, transcripts_path: str | Path, *, out: str | Path | None = None) -> dict:
    """Attach a transcript map to a whole bundle; write it to `out` (or in place) and report."""
    root = Path(bundle)
    transcripts = json.loads(Path(transcripts_path).read_text(encoding="utf-8"))
    used: set[str] = set()
    attached = 0
    voice = 0
    threads = 0
    for platform in PLATFORMS:
        directory = root / platform
        if not directory.is_dir():
            continue
        # Write in place unless an `out` is given: the caller copies the bundle first when it wants to
        # compare before replacing anything.
        target = Path(out) / platform if out else directory
        if out:
            target.mkdir(parents=True, exist_ok=True)
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            threads += 1
            attached += attach(payload, transcripts, used)
            voice += sum(1 for m in payload.get("messages") or [] for r in m.get("media") or [] if r.get("kind") == "audio")
            (target / path.name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    return {
        "threads": threads,
        "voice_notes": voice,
        "attached": attached,
        "without_transcript": voice - attached,
        "transcripts_unused": len(set(transcripts) - used),
        **({"out": str(out)} if out else {}),
    }
