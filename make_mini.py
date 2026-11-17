# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Carve a few conversations out of a donation bundle, unchanged, for a test run.

The output is a bundle: same layout, same filenames, byte-identical thread files. That matters —
`as_prompt`, message order and `media[].name` are exactly what the full build produced, so a chunk
built here has the same id and the same text as the same chunk in the full corpus.

Media bytes are **not** copied. Retrieval never reads them (it reads `media[].name`), so an index
built on this bundle behaves like the real one; only playback and STT need the bytes.

    uv run 26_THESIS/make_mini.py --bundle "26_THESIS/data/2_donation upload-2026-10-27" \\
        --out 26_THESIS/data/7_mini --thread fb:OTHER-004 --thread wa:OTHER-572
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

try:
    import typer
except ModuleNotFoundError:  # keep the script stdlib-only when typer is absent
    typer = None


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def main(bundle: Path, out: Path, threads: list[str], force: bool) -> None:
    src_report_path = bundle / "report.json"
    report = json.loads(src_report_path.read_text()) if src_report_path.exists() else {}

    wanted = []
    for spec in threads:
        platform, _, partner = spec.partition(":")
        if not partner:
            sys.exit(f"--thread wants platform:partner, got {spec!r}")
        path = bundle / platform / f"{partner}.json"
        if not path.exists():
            sys.exit(f"no such thread in the bundle: {path}")
        wanted.append((platform, partner, path))

    if out.exists() and any(out.iterdir()):
        if not force:
            sys.exit(f"{out} is not empty — pass --force to replace it")
        shutil.rmtree(out)

    entries, labels, media, senders = [], set(), Counter(), Counter()
    for platform, partner, path in wanted:
        data = json.loads(path.read_text())
        msgs = data["messages"]
        for m in msgs:
            labels.add(m.get("sender") or "")
            senders[m.get("sender") or "(none)"] += 1
            for md in m.get("media") or []:
                media[md.get("kind", "?")] += 1
        dest = out / platform / f"{partner}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        entries.append({
            "platform": platform, "partner": partner, "file": f"{platform}/{partner}.json",
            "sha256_16": sha(dest), "messages": len(msgs),
            "first": data.get("first"), "last": data.get("last"),
            "sources": data.get("sources"),
            "media_refs": sum(len(m.get("media") or []) for m in msgs),
            "voice_notes": sum(1 for m in msgs for md in (m.get("media") or []) if md.get("kind") == "audio"),
            "calls": sum(1 for m in msgs if m.get("call_s") is not None),
        })

    manifest = {
        "kind": "test subset of a donation bundle",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_bundle": str(bundle.resolve()),
        "source_report_generated_at": report.get("generated_at"),
        "render_version": report.get("render_version"),
        "threads": entries,
        "labels": sorted(l for l in labels if l),
        "senders": dict(senders),
        "media_refs_by_kind": dict(media),
        "media_bytes_copied": False,
        "note": ("Thread files are byte-identical copies; media bytes are omitted. Chunk ids and "
                 "as_prompt therefore match the full build. Labels are the full build's labels — "
                 "filtering happens here, after labelling, so nothing is renumbered."),
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    if src_report_path.exists():
        shutil.copy2(src_report_path, out / "report.json")

    (out / "README.md").write_text(f"""# test subset — {len(entries)} conversations

Carved from `{bundle.name}` at {manifest['built_at']}. Render version **{manifest['render_version']}**.
Thread files are byte-identical; **media bytes are not here** (retrieval reads the names only).

| thread | messages | voice notes | calls |
|---|---|---|---|
""" + "".join(
        f"| {e['platform']} {e['partner']} | {e['messages']:,} | {e['voice_notes']} | {e['calls']} |\n"
        for e in entries) + f"""
Labels present: {', '.join(f'`{l}`' for l in manifest['labels'])}.

```
# lexical index, no model, seconds
uv run 26_THESIS/retrieval/retrieval/cli.py build --bundle {out} --kind lexical \\
    --out 26_THESIS/retrieval/out/mini-lex --donor-id mini-test
# vector index, needs an embedding endpoint (see MANIFEST / config drift note)
uv run 26_THESIS/retrieval/retrieval/cli.py build --bundle {out} --kind vector \\
    --out 26_THESIS/retrieval/out/mini-vec --donor-id mini-test
```
""")

    print(f"threads   : {len(entries)}")
    for e in entries:
        print(f"  {e['platform']:3} {e['partner']:10} {e['messages']:>5} msgs  "
              f"{e['voice_notes']:>3} voice  {e['calls']:>4} calls  {e['sha256_16']}")
    print(f"labels    : {', '.join(manifest['labels'])}")
    print(f"senders   : {dict(senders)}")
    print(f"media refs: {dict(media)} (bytes not copied)")
    print(f"written   : {out.resolve()}")


if typer is None:
    if __name__ == "__main__":
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--bundle", required=True, type=Path)
        ap.add_argument("--out", required=True, type=Path)
        ap.add_argument("--thread", action="append", required=True)
        ap.add_argument("--force", action="store_true")
        a = ap.parse_args()
        main(a.bundle, a.out, a.thread, a.force)
else:
    def _cli(bundle: Path, out: Path, thread: list[str], force: bool = False) -> None:
        main(bundle, out, thread, force)

    if __name__ == "__main__":
        typer.run(_cli)
