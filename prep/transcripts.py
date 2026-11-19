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


def attach(payload: dict, transcripts: dict, used: set[str]) -> int:
    """Set `transcript` (or `stt: unintelligible`) on every audio reference the map knows about.

    Two shapes are accepted: a flat `{name: text}` map from an older STT run, and the verdict records
    `prep/stt.py` writes (`{verdict, text, confidence}`). The second is preferred because a note the
    engine *listened to and rejected* is a fact worth keeping — silence in the record reads as a bug,
    and the renderer says `<voice unintelligible>` when it is told.
    """
    attached = 0
    for message in payload.get("messages") or []:
        for ref in message.get("media") or []:
            if ref.get("kind") != "audio":
                continue
            name = ref.get("name") or ""
            entry = transcripts.get(name)
            if isinstance(entry, str):
                entry = {"verdict": "transcribed" if entry else "unintelligible", "text": entry}
            if not isinstance(entry, dict):
                continue
            if entry.get("verdict") == "transcribed" and entry.get("text"):
                ref["transcript"] = entry["text"]
                if entry.get("confidence") is not None:
                    ref["stt_confidence"] = round(float(entry["confidence"]), 3)
                attached += 1
            elif entry.get("verdict") == "unintelligible":
                ref["stt"] = "unintelligible"
                attached += 1
            used.add(name)
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
