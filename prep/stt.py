"""Speech to text, with the decisions written where the repo can keep them.

The engine lives outside both repositories (`stt_lean/donation`, adapters over FluidAudio, whisper.cpp,
ONNX and hosted APIs), and that is fine: trying four engines is a benchmark, not a pipeline. What is *not*
fine is the choice of model and the retry rule living in an untracked working tree, because they decide
what the corpus says. So the invocation lives here, versioned:

    fluidaudiocli transcribe <wav> --model-version ultra --encoder-precision int8 --language <hint>

and, when that returns nothing on a clip that may hold speech, a second attempt on a padded copy, taken
only above a confidence threshold.

**Three verdicts, and only three.** A voice note ends up `transcribed`, `unintelligible` (the engine
listened and had nothing to give), or `no_bytes` (the export referenced a file it did not ship). The
middle one matters: silence in the record reads like a bug or like the message had no content, and the
renderer says `<voice unintelligible>` so nobody has to guess which.

Measured on this corpus (874 notes, 8.77 h):
    v3    → 17 empty, 308 s on the old CLI build
    ultra → 10 empty, 826 s (spawn + model load dominates: 874 launches × ~0.25 s)
The retry recovers the boundary-trimmed ones; the threshold rejects the inventions ("patrón",
"And" on one-second clips, both at confidence 0.42 against 0.68–0.97 for real speech).
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

DEFAULT_CLI = Path.home() / ".local/share/FluidAudio/.build/release/fluidaudiocli"
MODEL_VERSION = "ultra"
ENCODER_PRECISION = "int8"
PAD_SECONDS = 0.4
# Below this, a padded retry is treated as invention rather than speech. Real speech measured 0.68–0.97,
# hallucinations 0.42 — the gap is wide enough that the exact value is not load-bearing.
PAD_RETRY_CONFIDENCE = 0.6

PLATFORMS = ("fb", "ig", "wa")


def _probe(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out) if out else None
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _to_wav(source: Path, target: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-ar", "16000", "-ac", "1",
             "-c:a", "pcm_s16le", str(target)],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _padded(wav: Path, target: Path) -> bool:
    pad = f"adelay={int(PAD_SECONDS * 1000)}|{int(PAD_SECONDS * 1000)},apad=pad_dur={PAD_SECONDS}"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", pad,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(target)],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _transcribe(cli: Path, wav: Path, language: str | None, scratch: Path) -> dict | None:
    out = scratch / "asr.json"
    cmd = [str(cli), "transcribe", str(wav), "--word-timestamps", "--output-json", str(out),
           "--model-version", MODEL_VERSION, "--encoder-precision", ENCODER_PRECISION]
    if language:
        cmd += ["--language", language]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode != 0 or not out.exists():
        return None
    return json.loads(out.read_text(encoding="utf-8"))


def transcribe_clip(cli: Path, source: Path, language: str | None) -> dict:
    """One voice note. Returns `{verdict, text, confidence, seconds}` — never raises for bad audio."""
    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td)
        wav = scratch / "in.wav"
        if not _to_wav(source, wav):
            return {"verdict": "no_bytes", "text": "", "confidence": 0.0, "seconds": None}
        seconds = _probe(wav)
        first = _transcribe(cli, wav, language, scratch)
        if first is None:
            return {"verdict": "no_bytes", "text": "", "confidence": 0.0, "seconds": seconds}

        text = (first.get("text") or "").strip()
        confidence = float(first.get("confidence") or 0.0)
        if text:
            return {"verdict": "transcribed", "text": text, "confidence": confidence, "seconds": seconds}

        # Empty on something that may well hold speech: the sliding window trims a short clip to nothing.
        padded = scratch / "padded.wav"
        if _padded(wav, padded):
            second = _transcribe(cli, padded, language, scratch)
            if second:
                retry_text = (second.get("text") or "").strip()
                retry_conf = float(second.get("confidence") or 0.0)
                if retry_text and retry_conf >= PAD_RETRY_CONFIDENCE:
                    return {"verdict": "transcribed", "text": retry_text, "confidence": retry_conf, "seconds": seconds}
        return {"verdict": "unintelligible", "text": "", "confidence": confidence, "seconds": seconds}


def run(bundle: str | Path, out: str | Path, *, language: str | None = None,
        cli: str | Path | None = None, limit: int | None = None) -> dict:
    """Every voice note in a bundle. Writes `{stored_name: {verdict, text, confidence, seconds}}`."""
    root = Path(bundle)
    binary = Path(cli or os.environ.get("FLUIDAUDIO_CLI", DEFAULT_CLI))
    if not binary.exists():
        raise SystemExit(f"fluidaudiocli not found at {binary} (set FLUIDAUDIO_CLI)")

    results: dict[str, dict] = {}
    missing = Path("")
    count = 0
    for platform in PLATFORMS:
        directory = root / platform
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for message in payload.get("messages") or []:
                for ref in message.get("media") or []:
                    if ref.get("kind") != "audio":
                        continue
                    name = ref.get("name") or ""
                    if not name or name in results:
                        continue
                    if limit is not None and count >= limit:
                        break
                    source = root / "media" / name
                    count += 1
                    if ref.get("missing") or not source.exists():
                        results[name] = {"verdict": "no_bytes", "text": "", "confidence": 0.0, "seconds": None}
                        continue
                    results[name] = transcribe_clip(binary, source, language)

    Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    verdicts: dict[str, int] = {}
    for value in results.values():
        verdicts[value["verdict"]] = verdicts.get(value["verdict"], 0) + 1
    return {"notes": len(results), "verdicts": verdicts, "out": str(out), **({"unused": str(missing)} if missing.name else {})}
