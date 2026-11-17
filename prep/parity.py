"""Does the reference renderer agree with the bundle, message by message?

`prep/render.py` exists so a register change is a re-render and not a re-upload. That promise is only
worth something if the two renderers agree *today*, on every message, including the ugly ones — so this
runs the Python renderer over a built bundle and compares byte for byte against `as_prompt`.

    uv run python -B -m prep parity --bundle "…/2_donation upload-2026-10-27"

Exit code is the answer: 0 when every message matches, 1 when any does not (with the first few shown).
A drift here is a bug in this file or in `minimal.ts`, never a matter of taste — which is why it is a
test and not a discussion.
"""
from __future__ import annotations

import json
from pathlib import Path

from .render import render_thread

PLATFORMS = ("fb", "ig", "wa")


def threads(bundle: str | Path):
    """Every conversation in a bundle, as `(platform, path, payload)`."""
    root = Path(bundle)
    for platform in PLATFORMS:
        directory = root / platform
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            yield platform, path, json.loads(path.read_text(encoding="utf-8"))


def compare(bundle: str | Path, *, show: int = 5, limit: int | None = None) -> dict:
    """Render every message and compare against the cache. Returns the counts and the first failures."""
    failures: list[dict] = []
    checked = 0
    mismatched = 0
    for platform, path, payload in threads(bundle):
        messages = payload.get("messages") or []
        rendered = render_thread(messages)
        for record, line in zip(messages, rendered):
            checked += 1
            cached = record.get("as_prompt", "")
            if line != cached:
                mismatched += 1
                if len(failures) < show:
                    failures.append(
                        {
                            "file": str(path.relative_to(bundle)),
                            "ts_ms": record.get("ts_ms"),
                            "expected": cached,
                            "rendered": line,
                        }
                    )
        if limit is not None and checked >= limit:
            break
    return {"checked": checked, "mismatched": mismatched, "failures": failures}
