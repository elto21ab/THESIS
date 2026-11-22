"""`26_THESIS/config.yaml`, read once, validated, and handed to the stages that use it.

The deid side has the same arrangement. Two repos, two files, and an asserted link between them: the
render version here must match `RENDER_VERSION` in `retrieval/retrieval/chunk.py` and in
`thesis-deid/src/core/standard/minimal.ts`, because chunk ids are keyed on it and a mismatch produces an
index that names a format it does not contain.

Unknown keys are an error, not a warning: a misspelt knob that is silently ignored is how a decision ends
up living in the code instead of in the file.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "config.yaml"

REQUIRED = {
    "render": ("version", "unintelligible"),
    "merge": ("min_overlap", "min_shared_messages"),
    "stt": ("model_version", "encoder_precision", "pad_seconds", "pad_retry_confidence"),
    "languages": (
        "min_confidence",
        "min_context_chars",
        "min_message_chars",
        "min_votes",
        "vote",
        "apply",
        "non_english_min_share",
        "english_min_share",
    ),
}


def load(path: str | Path | None = None) -> dict:
    text = Path(path or SOURCE).read_text(encoding="utf-8")
    parsed = yaml.safe_load(text) or {}
    problems: list[str] = []
    for section, keys in REQUIRED.items():
        block = parsed.get(section)
        if not isinstance(block, dict):
            problems.append(f"missing section: {section}")
            continue
        for key in keys:
            if key not in block:
                problems.append(f"missing key: {section}.{key}")
        for key in block:
            if key not in keys:
                problems.append(f"unknown key: {section}.{key}")
    for section in parsed:
        if section not in REQUIRED:
            problems.append(f"unknown section: {section}")
    if problems:
        raise SystemExit("config.yaml: " + "; ".join(problems))
    return parsed


CONFIG = load()


def render_version() -> str:
    from .render import RENDER_VERSION

    configured = CONFIG["render"]["version"]
    if configured != RENDER_VERSION:
        raise SystemExit(
            f"config.yaml says render {configured}, prep/render.py says {RENDER_VERSION} — one of them is stale"
        )
    return configured


def chunk_version() -> str:
    """The chunker's own constant, asserted against this file (it decides chunk ids)."""
    from retrieval.retrieval.chunk import RENDER_VERSION as CHUNK_VERSION

    if CONFIG["render"]["version"] != CHUNK_VERSION:
        raise SystemExit(
            f"config.yaml says render {CONFIG['render']['version']}, retrieval/chunk.py says {CHUNK_VERSION}"
        )
    return CHUNK_VERSION
