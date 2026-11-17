# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Transcribe a bundle's voice notes with Parakeet (FluidAudio CLI) into one JSON map.

    uv run 26_THESIS/transcribe.py --bundle "26_THESIS/data/8_corpus-2026-11-17" \\
        --out "26_THESIS/data/8_corpus-2026-11-17.transcripts.json" --jobs 4

Why this shape:

* **Keyed by the stored media name** (`wa001-00000605-AUDIO.opus`), because that is the only stable
  handle the bundle keeps. The export's own name is gone by then, and the message index would break
  on the next rebuild.
* **Resumable.** The wav and the per-file JSON are cached; a run that dies at file 700 of 876 costs
  the seven minutes, not the whole corpus. The map is rewritten every 25 files for the same reason.
* **16 kHz mono wav** in a scratch dir: Parakeet is trained at 16 kHz, and the bundle holds six
  containers (`aac`, `mp4`, `ogg`, `opus`, `wav`, `mp3`) that AVFoundation does not all decode.
  ffmpeg does, once, and the conversion is 20 ms per file.
* English/Danish/Norwegian/Swedish/German are all in parakeet-tdt v3's training set, which is what
  these chats are (Danish-English code-mixing included). It is not a translator; it transcribes
  what it hears, in the language it hears.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def duration_of(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def to_wav(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 44:
        return True
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        capture_output=True, text=True)
    return out.returncode == 0 and dst.exists() and dst.stat().st_size > 44


def parse_json(payload: dict) -> str:
    """FluidAudio's JSON has changed shape across versions; take the first field that holds words."""
    for key in ("text", "transcription", "transcript"):
        if isinstance(payload.get(key), str):
            return payload[key].strip()
    parts = []
    for seg in payload.get("segments") or []:
        if isinstance(seg, dict) and isinstance(seg.get("text"), str):
            parts.append(seg["text"].strip())
    return " ".join(p for p in parts if p).strip()


def batch(sources: list[Path], work: Path, model: str, timeout: float) -> dict[str, tuple[str, str]]:
    """Transcribe one batch, returning {name: (text, note)}.

    Batched because model load dominates: a 20-second voice note costs ~0.3 s to decode and ~3 s to
    load Parakeet. Fifty files per invocation amortises that to noise.
    """
    out: dict[str, tuple[str, str]] = {}
    ready: list[Path] = []
    for src in sources:
        wav = work / (src.stem + ".wav")
        if not to_wav(src, wav):
            out[src.name] = ("", "convert-failed")
            continue
        ready.append(wav)
    if not ready:
        return out

    dest = work / "out"
    dest.mkdir(exist_ok=True)
    cmd = ["uvx", "--from", "parakeet-mlx", "parakeet-mlx", *map(str, ready),
           "--model", model, "--output-format", "json", "--output-dir", str(dest)]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        for wav in ready:
            out[wav.stem + ".bin"] = ("", "timeout")
        return out
    for wav in ready:
        name = next((s.name for s in sources if s.stem == wav.stem), wav.stem)
        js = dest / (wav.stem + ".json")
        if not js.exists():
            tail = (run.stderr or run.stdout).strip().splitlines()[-1:] or ["no output"]
            out[name] = ("", f"failed: {tail[0][:60]}")
            continue
        try:
            out[name] = (parse_json(json.loads(js.read_text())), "ok")
        except json.JSONDecodeError:
            out[name] = ("", "bad-json")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="mlx-community/parakeet-tdt-0.6b-v3")
    ap.add_argument("--work", type=Path, default=Path(tempfile.gettempdir()) / "stt-wav")
    ap.add_argument("--per-call", type=int, default=40, help="files per model load")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", help="only media names matching this regex")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--keep-wav", action="store_true")
    a = ap.parse_args()

    if not shutil.which("uvx"):
        sys.exit("uvx not found — install uv")
    media = a.bundle / "media"
    if not media.is_dir():
        sys.exit(f"no media/ under {a.bundle}")
    files = sorted(p for p in media.iterdir() if p.is_file())
    if a.match:
        pat = re.compile(a.match)
        files = [p for p in files if pat.search(p.name)]
    if a.limit:
        files = files[:a.limit]

    a.work.mkdir(parents=True, exist_ok=True)
    done: dict[str, str] = {}
    if a.out.exists():
        done = json.loads(a.out.read_text())

    t0 = time.time()
    total = sum(duration_of(p) for p in files)
    print(f"{len(files)} files, {total/3600:.2f} h of audio, {a.per_call} per call", flush=True)

    todo = [p for p in files if p.name not in done]
    notes: dict[str, int] = {}
    processed = 0
    for i in range(0, len(todo), a.per_call):
        part = todo[i:i + a.per_call]
        for name, (text, note) in batch(part, a.work, a.model, a.timeout).items():
            notes[note] = notes.get(note, 0) + 1
            if text:
                done[name] = text
            processed += 1
        a.out.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n")
        el = time.time() - t0
        print(f"  {processed}/{len(todo)}  {el:.0f}s  {notes}", flush=True)
    a.out.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n")

    if not a.keep_wav:
        shutil.rmtree(a.work, ignore_errors=True)

    empty = sum(1 for p in files if p.name not in done)
    print(f"\ntranscribed {len(done)}/{len(files)}  empty-or-failed {empty}  "
          f"{time.time()-t0:.0f}s  ({total/3600:.2f} h audio → "
          f"{total/max(1e-9, time.time()-t0):.0f}× realtime)")
    print(f"written {a.out}")
    if notes:
        print(f"notes   {notes}")


if __name__ == "__main__":
    main()
