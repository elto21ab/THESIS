"""
ASR Benchmark: Compare A (WhisperX), B (Parakeet ONNX), D (transformers)
─────────────────────────────────────────────────────────────────────────
Run:  .venv-asr/bin/python tools/benchmark_asr.py
      .venv-asr/bin/python tools/benchmark_asr.py --tests A B    # subset
      .venv-asr/bin/python tools/benchmark_asr.py --audio my.wav

Outputs:
  - Console comparison table
  - benchmark_results.json  (all raw outputs)
  - benchmark_summary.md    (human-readable)

Test C (parakeet.cpp) is NOT run here — it requires a compiled binary.
See notes at the bottom of this file for how to set it up.
"""

import argparse
import json
import time
import sys
from pathlib import Path

AUDIO_PATH = Path("asr_target.wav")


# ─────────────────────────────────────────────────────────────────────────────
def run_test_A(audio_path: Path) -> dict:
    """Test A: WhisperX (faster-whisper CTranslate2 + pyannote)."""
    print("\n" + "═" * 60)
    print("  TEST A — WhisperX / faster-whisper large-v3")
    print("═" * 60)

    import os, warnings, torch
    os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:pyannote")
    warnings.filterwarnings("ignore", message="torchcodec is not installed")
    warnings.filterwarnings("ignore", message="Lightning automatically upgraded")

    import whisperx
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers

    HF_TOKEN  = os.environ.get("HF_TOKEN")
    DEVICE_PT = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()

    print(f"[A] loading whisper large-v3…")
    model = whisperx.load_model("large-v3", device="cpu", compute_type="int8",
                                 language="en")
    t_load = time.perf_counter()

    print(f"[A] transcribing…")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=16)
    lang = result.get("language", "en")

    align_model, meta = whisperx.load_align_model(language_code=lang, device=DEVICE_PT)
    result = whisperx.align(result["segments"], align_model, meta, audio, DEVICE_PT)
    t_transcribe = time.perf_counter()

    print(f"[A] diarizing…")
    diarize_model = DiarizationPipeline(
        model_name="pyannote/speaker-diarization-community-1",
        use_auth_token=HF_TOKEN,
        device=torch.device(DEVICE_PT),
    )
    diarize_segs = diarize_model(str(audio_path), max_speakers=2)
    result = assign_word_speakers(diarize_segs, result)
    t_diarize = time.perf_counter()

    segments = result.get("segments", [])
    transcript_lines = []
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        print(f"  [{_fmt(seg['start'])}–{_fmt(seg['end'])}] {spk}: {seg['text'].strip()}")
        transcript_lines.append({"start": seg["start"], "end": seg["end"],
                                  "speaker": spk, "text": seg["text"].strip()})

    dur = _audio_duration(audio_path)
    timings = {
        "load_asr_s":    round(t_load - t0, 2),
        "transcribe_s":  round(t_transcribe - t_load, 2),
        "diarize_s":     round(t_diarize - t_transcribe, 2),
        "total_s":       round(t_diarize - t0, 2),
        "audio_duration_s": round(dur, 2),
        "rtf":           round((t_diarize - t0) / dur, 3),
    }
    _print_timings(timings)
    return {"method": "A-whisperx", "model": "whisper-large-v3",
            "timings": timings, "transcript": transcript_lines}


# ─────────────────────────────────────────────────────────────────────────────
def run_test_B(audio_path: Path) -> dict:
    """Test B: Parakeet-TDT-v3 via onnx-asr + pyannote."""
    print("\n" + "═" * 60)
    print("  TEST B — Parakeet-TDT-0.6b-v3 (ONNX Runtime)")
    print("═" * 60)

    # Import inline to avoid cross-contamination
    sys.path.insert(0, str(Path(__file__).parent))
    import transcript_parakeet_onnx as b
    b.AUDIO_PATH = audio_path
    return b.run(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
def run_test_D(audio_path: Path, model_id: str = "openai/whisper-large-v3-turbo") -> dict:
    """Test D: HuggingFace transformers pipeline + pyannote."""
    print("\n" + "═" * 60)
    print(f"  TEST D — transformers / {model_id}")
    print("═" * 60)

    sys.path.insert(0, str(Path(__file__).parent))
    import transcript_transformers as d
    d.AUDIO_PATH = audio_path
    return d.run(audio_path, model_id=model_id)


# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(results: list[dict]) -> None:
    """Print a side-by-side timing + quality table."""
    print("\n" + "═" * 70)
    print("  BENCHMARK SUMMARY")
    print("═" * 70)
    print(f"  {'Method':<30} {'Load':>8} {'ASR':>8} {'Diarize':>8} {'Total':>8} {'RTF':>6}")
    print("  " + "─" * 68)
    for r in results:
        t = r["timings"]
        print(f"  {r['method']:<30} {t['load_asr_s']:>7.1f}s {t['transcribe_s']:>7.1f}s "
              f"{t['diarize_s']:>7.1f}s {t['total_s']:>7.1f}s {t['rtf']:>5.2f}×")
    print("  " + "─" * 68)
    print(f"  Audio duration: {results[0]['timings']['audio_duration_s']}s")
    print("  RTF < 1.0 = faster than real time")
    print("═" * 70)

    # Show first line of each transcript for quick quality check
    print("\n  FIRST SEGMENT COMPARISON (for WER spot-check):")
    print("  " + "─" * 68)
    for r in results:
        segs = r.get("transcript", [])
        first = segs[0] if segs else {}
        text = first.get("text") or first.get("word", "")
        print(f"  [{r['method'][:28]}]: {text[:60]}")
    print("═" * 70)


def write_summary_md(results: list[dict], out_path: Path) -> None:
    lines = ["# ASR Benchmark Results\n"]
    lines.append("| Method | Load | ASR | Diarize | Total | RTF |")
    lines.append("|--------|-----:|----:|--------:|------:|----:|")
    for r in results:
        t = r["timings"]
        lines.append(f"| {r['method']} | {t['load_asr_s']}s | {t['transcribe_s']}s | "
                     f"{t['diarize_s']}s | {t['total_s']}s | {t['rtf']}× |")
    lines.append(f"\nAudio duration: {results[0]['timings']['audio_duration_s']}s\n")
    lines.append("\n## Transcripts\n")
    for r in results:
        lines.append(f"### {r['method']} ({r['model']})\n")
        for seg in r.get("transcript", [])[:20]:
            t_start = seg.get("start", 0)
            t_end   = seg.get("end", 0)
            spk     = seg.get("speaker", "")
            text    = seg.get("text") or seg.get("word", "")
            lines.append(f"[{_fmt(t_start)}–{_fmt(t_end)}] **{spk}**: {text}")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[saved] summary → {out_path}")


def _fmt(s) -> str:
    s = int(s or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def _audio_duration(path: Path) -> float:
    import wave
    with wave.open(str(path)) as wf:
        return wf.getnframes() / wf.getframerate()


def _print_timings(t: dict) -> None:
    print(f"\n⏱  Load: {t['load_asr_s']}s | ASR: {t['transcribe_s']}s | "
          f"Diarize: {t['diarize_s']}s | Total: {t['total_s']}s | RTF: {t['rtf']}×")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ASR benchmark runner")
    parser.add_argument("--audio", default=str(AUDIO_PATH),
                        help="Path to 16kHz WAV file")
    parser.add_argument("--tests", nargs="+", default=["A", "B", "D"],
                        choices=["A", "B", "D"],
                        help="Which tests to run (default: A B D)")
    parser.add_argument("--model-d", default="openai/whisper-large-v3-turbo",
                        help="Model ID for Test D (transformers)")
    args = parser.parse_args()

    audio = Path(args.audio)
    if not audio.exists():
        print(f"[!] Audio file not found: {audio}")
        print("    Run transcript_whisperx.py first to download the test audio.")
        sys.exit(1)

    results = []
    test_fns = {"A": run_test_A, "B": run_test_B,
                "D": lambda p: run_test_D(p, args.model_d)}

    for t in args.tests:
        try:
            r = test_fns[t](audio)
            results.append(r)
        except Exception as e:
            print(f"\n[!] Test {t} failed: {e}")
            import traceback; traceback.print_exc()

    if results:
        print_comparison_table(results)
        Path("benchmark_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False))
        write_summary_md(results, Path("benchmark_summary.md"))


# ─────────────────────────────────────────────────────────────────────────────
# TEST C — parakeet.cpp (Metal, M2 only)
# ─────────────────────────────────────────────────────────────────────────────
# parakeet.cpp is a C++ binary — it does NOT install via pip.
# It uses GGML + Metal and is the fastest option on M2.
#
# Install steps (macOS, M2):
#   git clone https://github.com/Frikallo/parakeet.cpp
#   cd parakeet.cpp && mkdir build && cd build
#   cmake .. -DGGML_METAL=ON && make -j$(sysctl -n hw.physicalcpu)
#   ./bin/parakeet --model parakeet-tdt-0.6b-v3-Q8_0.gguf --file asr_target.wav
#
# Then call it from Python via subprocess:
#   import subprocess, json
#   result = subprocess.run(
#       ["./parakeet.cpp/build/bin/parakeet", "--model", "...", "--file", "asr_target.wav",
#        "--output-json"],
#       capture_output=True, text=True
#   )
#   data = json.loads(result.stdout)
#
# Diarization: parakeet.cpp has built-in Sortformer diarization (≤4 speakers).
# Enable it with --diarize flag. No pyannote needed.
#
# GGML quantization options (tradeoff: size/speed vs accuracy):
#   Q4_K_M  ~45% size, fastest,   ~1-2% WER increase
#   Q8_0    ~75% size, fast,      near-lossless
#   F16     100% size, reference, full accuracy
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
