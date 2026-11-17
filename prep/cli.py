"""`uv run python -B -m prep parity|render|merge ...`

Three verbs, one purpose: do on the server what the donor tool already did, so that fixing a rule does
not mean asking somebody for their export again.

    parity       compare the reference renderer against the bundle's cached `as_prompt` (a test)
    render       write `as_prompt` again from the fields the bundle holds (a rebuild)
    merge        re-merge the parts a `--keep-all` bundle kept, on content rather than names
    transcripts  attach an STT pass's `{stored_name: text}` map to `media[].transcript`

Nothing here reads the raw export: that is the point. A bundle is enough.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .canonical import canonicalise
from .merge import merge_payloads
from .render import render_thread

PLATFORMS = ("fb", "ig", "wa")


def _threads(root):
    from pathlib import Path

    for platform in PLATFORMS:
        directory = Path(root) / platform
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            yield platform, path, json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prep", description=__doc__)
    parser.add_argument("verb", choices=("parity", "render", "merge", "transcripts"))
    parser.add_argument("--bundle", required=True, help="a built donation bundle")
    parser.add_argument("--out", help="where `render` and `merge` write (default: print a summary)")
    parser.add_argument("--show", type=int, default=5, help="parity: how many mismatches to print")
    parser.add_argument("--transcripts", help="transcripts: a {stored_name: text} JSON map from the STT pass")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.verb == "parity":
        from .parity import compare

        result = compare(args.bundle, show=args.show)
        ok = result["mismatched"] == 0
        if args.json:
            print(json.dumps(result, indent=1))
        else:
            matched = result["checked"] - result["mismatched"]
            print(
                f"parity  render {_version()}  {matched}/{result['checked']} match"
                + ("" if ok else f", {result['mismatched']} differ")
            )
            for failure in result["failures"]:
                print(f"\n  {failure['file']} @ {failure['ts_ms']}")
                print(f"    cached:   {failure['expected']!r}")
                print(f"    rendered: {failure['rendered']!r}")
        return 0 if ok else 1

    if args.verb == "transcripts":
        from .transcripts import apply as attach_transcripts

        if not args.transcripts:
            raise SystemExit("transcripts: --transcripts <file.json> is required")
        print(json.dumps(attach_transcripts(args.bundle, args.transcripts, out=args.out), indent=1))
        return 0

    if args.verb == "render":
        written = 0
        changed = 0
        for _platform, path, payload in _threads(args.bundle):
            messages = payload.get("messages") or []
            for record, line in zip(messages, render_thread(messages)):
                written += 1
                if record.get("as_prompt") != line:
                    record["as_prompt"] = line
                    changed += 1
            # In place unless `--out` is given: the caller copies the bundle when it wants to compare
            # the two before replacing anything (a rebuild nobody can diff is not a rebuild).
            target = Path(path) if not args.out else Path(args.out) / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"messages": written, "changed": changed, "out": args.out or "in place"}))
        return 0

    merged = canonicalise(args.bundle, out=args.out, merge=merge_payloads)
    print(json.dumps(merged, indent=1))
    return 0


def _version() -> str:
    from .render import RENDER_VERSION

    return RENDER_VERSION


if __name__ == "__main__":
    sys.exit(main())
