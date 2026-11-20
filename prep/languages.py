"""Which language each voice note is probably in, decided from the words around it.

The engine takes one `--language` code and uses it as a decoder hint. Hard-coding Danish is wrong for a
donation from another country; hard-coding nothing throws away a signal. So the hint is *derived* — from
the text messages either side of the note, which are free and already in hand — and omitted when the
surrounding words do not agree.

    note          the audio to transcribe
    context       the ±N text messages around it, which is the only evidence available before STT
    verdict       pass the code, or pass nothing

Deliberately not audio language ID: that is a second model, a second download, and it answers a question
the text usually answers better (a shared link in English inside a Danish thread is still a Danish
conversation).

    uv run python -B -m prep languages --bundle … [--write]     # what it would tell the engine

The output is `{media_name: {"language": "da"|None, "confidence": float, "evidence": "…"}}`, and
`prep stt` reads it when `--languages` is given.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CONFIG

# The languages this corpus plausibly holds. `lingua` is used rather than a hand-rolled stopword table
# because Scandinavian function words overlap almost completely — a first attempt scored Danish 13,
# Norwegian 10 for one sentence and could never reach a verdict. A detector that models the languages
# properly is the difference between a hint and noise.
CODES = {
    "da": "DANISH", "en": "ENGLISH", "sv": "SWEDISH", "no": "NORWEGIAN", "de": "GERMAN",
    "es": "SPANISH", "fr": "FRENCH", "it": "ITALIAN", "nl": "DUTCH", "pt": "PORTUGUESE",
}

_DETECTOR = None


def _detector():
    global _DETECTOR
    if _DETECTOR is None:
        from lingua import Language, LanguageDetectorBuilder

        languages = [getattr(Language, name) for name in CODES.values() if hasattr(Language, name)]
        # Low accuracy mode is enough for whole threads and ~30x faster than the default.
        _DETECTOR = LanguageDetectorBuilder.from_languages(*languages).with_low_accuracy_mode().build()
    return _DETECTOR


def decide(context: str) -> tuple[str | None, float, str]:
    """A language code when the detector is confident, else None.

    Confidence is the winner's probability. A hint steers the decoder, so the bar is deliberately high:
    below it, the engine gets no hint at all and falls back on its multilingual default.
    """
    text = context.strip()
    if len(text) < CONFIG["languages"]["min_context_chars"]:
        # Too little evidence: a two-word exchange cannot separate Danish from Norwegian, and a wrong
        # hint is worse than none.
        return None, 0.0, f"only {len(text)} chars of context"
    detection = _detector().detect_language_of(text)
    if detection is None:
        return None, 0.0, "no detection"
    code = next((c for c, n in CODES.items() if n == detection.name), None)
    confidence = _detector().compute_language_confidence(text, detection)
    if code is None or confidence < CONFIG["languages"]["min_confidence"]:
        return None, round(confidence, 3), f"below threshold ({detection.name} {confidence:.2f})"
    return code, round(confidence, 3), f"{detection.name} {confidence:.2f}"


def _context(messages: list[dict], index: int, window: int = 6) -> str:
    """The messages either side of a voice note, whose speech is not available yet."""
    out = []
    for offset in range(1, window + 1):
        for j in (index - offset, index + offset):
            if 0 <= j < len(messages):
                text = messages[j].get("text") or ""
                if text.strip():
                    out.append(text)
    return " ".join(out)


def thread_language(thread_messages: list[dict]) -> tuple[str | None, float, str]:
    """One language for a whole conversation, by majority over its text messages.

    More robust than a per-note verdict: a single English link in a Danish thread does not change what
    the conversation is in, and a note whose neighbours are short has the whole thread to lean on. The
    vote is weighted by characters, because ten "ok" and one paragraph are not evidence of equal size.
    """
    votes: dict[str, int] = {}
    for message in thread_messages:
        text = (message.get("text") or "").strip()
        if len(text) < 8:
            continue
        code, confidence, _ = decide(text)
        if code is None:
            continue
        votes[code] = votes.get(code, 0) + int(len(text) * confidence)
    if not votes:
        return None, 0.0, "no votes"
    code, weight = max(votes.items(), key=lambda kv: kv[1])
    total = sum(votes.values())
    share = weight / total
    if share < 0.7:
        return None, round(share, 3), f"split thread ({votes})"
    return code, round(share, 3), f"{weight}/{total} characters"


def run(bundle: str | Path, out: str | Path | None = None, *, vote: str | None = None) -> dict:
    """One verdict per voice note, from the text around it — or one per thread when `vote=thread`."""
    root = Path(bundle)
    verdicts: dict[str, dict] = {}
    mode = vote or CONFIG["languages"]["vote"]
    for path in sorted(root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        messages = payload.get("messages") or []
        thread_code, thread_conf, thread_evidence = thread_language(messages)
        for index, message in enumerate(messages):
            for ref in message.get("media") or []:
                if ref.get("kind") != "audio" or not ref.get("name"):
                    continue
                if mode == "thread":
                    # The thread's answer, with the note's own neighbourhood as a tiebreak when the
                    # thread is genuinely mixed: an English note inside a mostly English conversation
                    # should not be dragged back to the majority language.
                    local_code, local_conf, local_evidence = decide(_context(messages, index))
                    code = local_code if (thread_code is None and local_code) else thread_code
                    confidence = thread_conf if code == thread_code else local_conf
                    evidence = f"thread: {thread_evidence}" if code == thread_code else f"local: {local_evidence}"
                else:
                    code, confidence, evidence = decide(_context(messages, index))
                verdicts[ref["name"]] = {
                    "language": code,
                    "confidence": round(confidence, 3),
                    "evidence": evidence,
                }
    counts: dict[str, int] = {}
    for value in verdicts.values():
        key = value["language"] or "none"
        counts[key] = counts.get(key, 0) + 1
    if out:
        Path(out).write_text(json.dumps(verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"notes": len(verdicts), "languages": counts, **({"out": str(out)} if out else {})}
