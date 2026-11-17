"""Turn a donation into the corpus the study reads: merge the parts, then write the bundle back out.

`canonicalise` is the recovery path. A donation built with `--keep-all` carries `_parts/` — one file per
read of a conversation — and `_dropped/` for everything a rule refused. Nothing in `fb/ ig/ wa/` needs
to change for a re-merge; this writes a new tree beside it, so the two can be compared before anything
is replaced.

What it does *not* do is decide anything the donor tool decides: labels stay as given, `as_prompt` stays
as cached (re-rendering is `prep render`, a separate verb with its own test). One stage, one decision,
so a diff says what changed.
"""
from __future__ import annotations

import json
from pathlib import Path

PLATFORMS = ("fb", "ig", "wa")


def _read(root: Path) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for platform in PLATFORMS:
        parts = root / "_parts" / platform
        if not parts.is_dir():
            continue
        for path in sorted(parts.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload.setdefault("platform", platform)
            out.append((path.stem, payload))
    return out


def canonicalise(bundle: str | Path, *, out: str | Path | None = None, merge=None) -> dict:
    """Re-merge a donation's parts. Returns the counts; writes the tree when `out` is given.

    Without `_parts/` there is nothing to redo, and this says so rather than pretending: a bundle built
    without `--keep-all` carries the merge's *result*, not its inputs.
    """
    root = Path(bundle)
    parts = _read(root)
    if not parts:
        return {
            "parts": 0,
            "threads": 0,
            "note": "no _parts/ in this bundle: built without --keep-all, so the merge cannot be redone "
            "from it. The input to the decision is what a bundle has to carry for a rebuild to be possible.",
        }

    groups = merge(parts)
    written = []
    if out:
        target = Path(out)
        for platform in PLATFORMS:
            (target / platform).mkdir(parents=True, exist_ok=True)
        for group in groups:
            payload = {
                "_note": "re-merged from _parts/ by prep.canonical; labels and as_prompt are the donation's",
                "platform": group["platform"],
                "participants": group["participants"],
                "sources": group["sources"],
                "first": min((m.get("ts_ms") or 0) for m in group["messages"]),
                "last": max((m.get("ts_ms") or 0) for m in group["messages"]),
                "messages": group["messages"],
            }
            name = f"{group['platform']}-{len(written) + 1:04d}.json"
            (target / group["platform"] / name).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            written.append(name)

    return {
        "parts": len(parts),
        "threads": len(groups),
        "multi_source": sum(1 for g in groups if len(g["sources"]) > 1),
        "messages": sum(len(g["messages"]) for g in groups),
        **({"out": str(out), "written": len(written)} if out else {}),
    }
