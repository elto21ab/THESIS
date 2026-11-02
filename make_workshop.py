#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Printable sheets for the supervisor workshop: the data→answer corridor, and its options.

Three A3 landscape sheets and one A3 portrait sheet, all measured on this donor's corpus:

  1  the corridor: counts per channel, and one message at each of the four levels
     (raw export → standardised entry → rendered text → chunk)
  2  chunking and retrieval: the flow, the knobs, and where a chunk boundary would fall
  3  edge cases: one real example per case per channel, at the rendered level
  4  (portrait) two long stretches, FB and WA, with chunk windows and their overlap drawn

    uv run 26_THESIS/make_workshop.py \\
        --bundle "26_THESIS/data/2_donation upload-2026-10-27" \\
        --readable "26_THESIS/data/1_export readable" \\
        --out "26_THESIS/data/6_workshop" [--pdf]
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
FB, WA = "OTHER-004", "OTHER-572"          # the two conversations greenlit for the print
# How many messages each portrait sheet carries. Fixed, not fitted: fb's lines are longer than wa's,
# so the same number does not fill the same space, and a number that runs long is easier to trim by
# hand than a page that runs short.
CHAT_MSGS = {("fb", FB): 290, ("wa", WA): 270}


def iso(ms: int | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(fmt)


def tokens(text: str) -> int:
    """The tool's estimator, ported so every figure agrees with report.json: JS `\\w` is ASCII, so a
    Danish letter counts as a symbol there — the same has to happen here."""
    if not text:
        return 0
    letters = sum(math.ceil(len(m) / 3.9) for m in re.findall(r"[A-Za-z_]+", text))
    digits = sum(math.ceil(len(m) / 1.6) for m in re.findall(r"\d+", text))
    symbols = re.findall(r"[^A-Za-z0-9_\s]", text)
    return letters + digits + (math.ceil(len(symbols) * 1.3) if symbols else 0)


def esc(s: str) -> str:
    return html.escape(s)


def load(bundle: Path) -> tuple[dict, list[dict]]:
    report = json.loads((bundle / "report.json").read_text())
    threads = []
    for path in sorted(bundle.glob("*/*.json")):
        if path.name == "report.json":
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        data["_path"] = str(path)
        threads.append(data)
    return report, threads


def thread_of(threads: list[dict], platform: str, partner: str) -> dict:
    return next(t for t in threads if t["platform"] == platform and t["partner"] == partner)


# ── the four levels, on one message ──────────────────────────────────────────
def raw_of(readable: Path, platform: str, partner: str) -> list[dict]:
    path = readable / platform / f"{partner}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text()).get("messages", [])


def level_rows(bundle_msg: dict, raw_msg: dict | None) -> dict[str, str]:
    """The same message at four levels of processing, as text for the sheet."""
    l1 = json.dumps(raw_msg, ensure_ascii=False, indent=1) if raw_msg else "(not in the export)"
    keep = {k: v for k, v in bundle_msg.items() if k != "as_prompt"}
    l2 = json.dumps(keep, ensure_ascii=False, indent=1)
    l3 = bundle_msg["as_prompt"]
    return {"L1 raw": l1, "L2 entry": l2, "L3 text": l3, "L4 chunk": ""}


def corpus_gradient(threads: list[dict], readable: Path) -> list[tuple[str, int, int]]:
    """Chars and tokens for the corpus at each level, so the ladder has a price tag.

    L1 is the *export's* object (bigger than the entry: `is_geoblocked_for_viewer` and friends), L2 the
    entry without the rendered string, L3 with it, L4 with the stamp stripped. Facebook and Instagram
    only — WhatsApp's export is already text, so it has no L1 to compare against.
    """
    counts = {"l1": [0, 0], "l2": [0, 0], "l3": [0, 0], "l4": [0, 0]}
    for platform in ("fb", "ig"):
        for path in sorted((readable / platform).glob("*.json")):
            for m in json.loads(path.read_text()).get("messages", []):
                blob = json.dumps(m, ensure_ascii=False)
                counts["l1"][0] += len(blob)
                counts["l1"][1] += tokens(blob)
    for t in threads:
        if t["platform"] not in ("fb", "ig"):
            continue
        for m in t["messages"]:
            entry = json.dumps({k: v for k, v in m.items() if k != "as_prompt"}, ensure_ascii=False)
            lean = re.sub(r"^\[[^\]]*\]\s*", "", m["as_prompt"])
            counts["l2"][0] += len(entry)
            counts["l2"][1] += tokens(entry)
            counts["l3"][0] += len(m["as_prompt"])
            counts["l3"][1] += tokens(m["as_prompt"])
            counts["l4"][0] += len(lean)
            counts["l4"][1] += tokens(lean)
    labels = [("L1 · raw export message", "l1"), ("L2 · standardised entry", "l2"),
              ("L3 · rendered text", "l3"), ("L4 · lean text, no stamp", "l4")]
    return [(label, counts[key][0], counts[key][1]) for label, key in labels]


def chunker():
    sys.path.insert(0, str(Path(__file__).parent / "retrieval"))
    from retrieval.bundle import load_bundle
    from retrieval.chunk import CHUNKERS, ChunkCfg

    return load_bundle, CHUNKERS, ChunkCfg


def chunk_stats(bundle: Path) -> list[dict]:
    load_bundle, CHUNKERS, ChunkCfg = chunker()
    threads, _, _ = load_bundle(bundle)
    rows = []
    for name, cfg, fn in (
        ("day", ChunkCfg(), CHUNKERS["day"]),
        ("fixed · target 512", ChunkCfg(target=512), CHUNKERS["fixed"]),
        ("fixed · target 256", ChunkCfg(target=256), CHUNKERS["fixed"]),
    ):
        chunks = [c for t in threads for c in fn(t, cfg, tokens)]
        sizes = sorted(c.n_msgs for c in chunks)
        rows.append({"name": name, "chunks": len(chunks), "tokens": sum(c.n_tokens for c in chunks),
                     "median": sizes[len(sizes) // 2], "min": sizes[0], "max": sizes[-1]})
    return rows


def spans_of(bundle: Path, platform: str, partner: str) -> list[tuple[int, int, int]]:
    """(i0, i1, seq) for one conversation under the day chunker — where the cuts would fall."""
    load_bundle, CHUNKERS, ChunkCfg = chunker()
    threads, _, _ = load_bundle(bundle)
    thread = next(t for t in threads if t.platform == platform and t.partner == partner)
    return [(c.i0, c.i1, c.seq) for c in CHUNKERS["day"](thread, ChunkCfg(), tokens)]


# ── edge cases ───────────────────────────────────────────────────────────────
CASES = [
    # The direction field is what separates a call that happened from the platform's bookkeeping:
    # "The video call ended, 55s" is an event with a duration, not an answered call.
    ("call, answered", "<call: in", lambda m: m.get("call_direction") in ("in", "out") and m.get("call_s", 0) > 0),
    ("call, missed", "missed", lambda m: m.get("call_missed")),
    ("call, event", "event", lambda m: m.get("call_direction") == "event"),
    ("voice note", "<voice", lambda m: "<voice" in m["as_prompt"]),
    ("photo / video burst", "x2", lambda m: "x2" in m["as_prompt"] or "<attached: VIDEO" in m["as_prompt"]),
    ("file · sticker · gif", "<attached: FILE", lambda m: any(k in m["as_prompt"] for k in ("<attached: FILE", "<attached: STICKER", "<attached: GIF"))),
    ("shared post", "<share:", lambda m: "<share:" in m["as_prompt"] and "post>" not in m["as_prompt"][-8:]),
    ("shared profile (no url)", "<share: post>", lambda m: "<share: post>" in m["as_prompt"]),
    ("reaction, with time", " at ", lambda m: "<react:" in m["as_prompt"] and " at " in m["as_prompt"]),
    ("reaction, sticker", "sticker by", lambda m: "sticker by" in m["as_prompt"]),
    ("deleted by sender", "<unsent>", lambda m: "<unsent>" in m["as_prompt"]),
    ("export left it empty", "<no content>", lambda m: "<no content>" in m["as_prompt"]),
    ("platform notice", "notice", lambda m: m.get("system") == "notice"),
    ("message deleted (notice)", "deleted", lambda m: m.get("system") == "deleted"),
    ("multiline body", "\\n", lambda m: "\n" in (m.get("text") or "").strip()),
    ("link in the text only", "<link:", lambda m: "<link:" in m["as_prompt"]),
]


def edge_table(threads: list[dict]) -> list[tuple[str, dict[str, str]]]:
    out = []
    for label, needle, test in CASES:
        cells = {}
        for platform in ("fb", "ig", "wa"):
            pool = (m for t in threads if t["platform"] == platform for m in t["messages"])
            hit = next((m for m in pool if test(m)), None)
            cells[platform] = hit["as_prompt"] if hit else "—"
        out.append((label, cells))
    return out


CSS_LANDSCAPE = """
@page { size: A3 landscape; margin: 9mm; }
body { font: 11.5px/1.32 "Helvetica Neue", Arial, sans-serif; color: #111; margin: 0; }
h1 { font-size: 21px; margin: 0 0 1mm; }
h1 small { font-weight: 400; font-size: 13px; color: #555; }
h2 { font-size: 12.5px; margin: 0 0 1.5mm; text-transform: uppercase; letter-spacing: .05em; color: #333; }
.grid { display: grid; gap: 5mm; }
.two { grid-template-columns: 1fr 1fr; }
.three { grid-template-columns: 1.15fr 1fr 1fr; }
.panel { border: 1px solid #b9b9b9; border-radius: 2.5mm; padding: 3.5mm; }
.wide { grid-column: 1 / -1; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 1mm 1.6mm; border-bottom: 1px solid #e0e0e0; vertical-align: top; }
th { font-weight: 600; font-size: 10.5px; color: #333; }
pre, code { font-family: "SF Mono", Menlo, monospace; font-size: 9.6px; }
pre { background: #f6f6f6; border-radius: 1.5mm; padding: 2mm 2.5mm; margin: 0; white-space: pre-wrap; }
.note { color: #555; font-size: 10.5px; margin: 1.5mm 0 0; }
.k { display: inline-block; background: #eee; border-radius: 1.5mm; padding: 0 1.2mm; font-size: 9px; }
.arrow { text-align: center; font-size: 16px; color: #888; }
.flow { display: flex; align-items: stretch; gap: 0; }
.flow .step { flex: 1; border: 1px solid #b9b9b9; border-radius: 2mm; padding: 2.5mm; font-size: 10px; }
.flow .sep { align-self: center; padding: 0 1.5mm; color: #999; font-size: 15px; }
.board { font-size: 9px; }
.rowlbl { font-size: 9px; color: #666; margin: 1mm 0 0.4mm; }
.strip { display: flex; gap: 0.6mm; flex-wrap: wrap; }
.m { border: 1px solid #ccc; border-radius: 1mm; padding: 0.6mm 1mm; white-space: nowrap; overflow: hidden;
     text-overflow: ellipsis; max-width: 100%; font-family: "SF Mono", Menlo, monospace; background: #fff; }
.m.in { background: #e6efff; border-color: #9bb7e6; }
.m.ov { background: #ffd6d6; border-color: #d99; }
.m.hit { background: #d8f2d0; border-color: #7a7; }
.m.miss { background: #f2f2f2; color: #999; }
.m.merged { background: #d8f2d0; border-color: #7a7; white-space: normal; }
.m.ctx { background: #f6f6f6; color: #555; }
.m.v { background: #eef; border-color: #99c; }
.page { break-after: page; page-break-after: always; }
.page + .page { break-before: page; page-break-before: always; margin-top: 6mm; }
.page:last-child { break-after: auto; }
td.ok { color: #111; }
td.no { color: #bbb; }
"""

CSS_PORTRAIT = """
@page { size: A3 portrait; margin: 9mm; }
body { font: 11.5px/1.3 "Helvetica Neue", Arial, sans-serif; color: #111; margin: 0; }
h1 { font-size: 20px; margin: 0 0 1mm; }
h1 small { font-weight: 400; font-size: 12.5px; color: #555; }
h2 { font-size: 12px; margin: 0 0 1.5mm; text-transform: uppercase; letter-spacing: .05em; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 5mm; }
.panel { border: 1px solid #b9b9b9; border-radius: 2.5mm; padding: 3mm; }
.legend { font-size: 10px; color: #444; margin: 1mm 0 2mm; }
.page { break-after: page; page-break-after: always; }
.page + .page { break-before: page; page-break-before: always; }
.twocol { columns: 2; column-gap: 6mm; }
.twocol .line { break-inside: avoid; }
.sw { display: inline-block; width: 9px; height: 9px; border-radius: 1mm; vertical-align: middle; }
.line { font-family: "SF Mono", Menlo, monospace; font-size: 10px; white-space: pre-wrap; padding: 0.5mm 1.2mm; }
.c0 { background: #eef2f8; }
.ov { background: #ffd6d6; }
.cut { border-top: 1.8px dashed #c00; margin: 1mm 0 0.6mm; position: relative; }
.cut span { background: #c00; color: #fff; font-size: 8px; padding: 0 1.2mm; border-radius: 0 0 1.5mm 1.5mm; }
"""


def panel(title: str, body: str, cls: str = "") -> str:
    return f'<div class="panel {cls}"><h2>{title}</h2>{body}</div>'


MARK = {
    "who": "background:#ffe9a8",
    "time": "background:#cfe6ff",
    "react": "background:#ffd9ec",
    "att": "background:#d8f2d0",
    "text": "",
}


ANNOTATE = [
    (re.compile(r"\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}\]|\[[0-9]{2}:[0-9]{2}\]"), "time"),
    (re.compile(r"\b(?:S|O[0-9]+):"), "who"),
    (re.compile(r"&lt;react:[^&]*?&gt;"), "react"),
    (re.compile(r"&lt;attached:[^&]*?&gt;|&lt;voice[^&]*?&gt;"), "att"),
]


def annotate(escaped: str) -> str:
    """Colour the four things that travel down the ladder, on escaped HTML."""
    for pattern, kind in ANNOTATE:
        escaped = pattern.sub(lambda m: f'<mark style="{MARK[kind]}">{m.group(0)}</mark>', escaped, count=1 if kind == "time" else 0)
    return escaped


def hl(text: str, value: str, kind: str) -> str:
    """Wrap one value in a coloured mark, inside already-escaped text."""
    if not value:
        return text
    return text.replace(esc(value), f'<mark style="{MARK[kind]}">{esc(value)}</mark>', 1)


def page1(report: dict, threads: list[dict], readable: Path) -> str:
    vol = report["volume"]
    rows = "".join(
        f"<tr><td>{p}</td><td>{d['conversations']}</td><td>{d['messages_K']:.1f}k</td>"
        f"<td>{d['text']['chars_M']:.2f}M</td><td>{d['text']['tokens_M']:.2f}M</td>"
        f"<td>{d['prompt']['tokens_M']:.2f}M</td><td>{d['audio']['files']}</td></tr>"
        for p, d in report["platforms"].items()
    )
    grad = corpus_gradient(threads, readable)
    grad_rows = "".join(
        f"<tr><td>{name}</td><td>{chars/1e6:.2f}M</td><td>{tok/1e6:.2f}M</td>"
        f"<td>{tok/grad[0][2]*100:.0f}%</td></tr>" for name, chars, tok in grad
    )

    # one rich message, four levels
    demo = thread_of(threads, "fb", FB)
    rich = max(demo["messages"], key=lambda m: sum(
        k in m["as_prompt"] for k in ("<react:", "<attached:", "<voice", "<call:", "<share:", "<link", "<unsent>")))
    raws = {m.get("timestamp_ms"): m for m in raw_of(readable, "fb", FB)}
    raw = raws.get(rich["ts_ms"])
    l1 = json.dumps(raw, ensure_ascii=False, indent=1) if raw else "(not found)"
    l2 = json.dumps({k: v for k, v in rich.items() if k != "as_prompt"}, ensure_ascii=False, indent=1)
    l3 = rich["as_prompt"]
    spans = spans_of(Path(report["_bundle"]), "fb", FB)
    span = next((s for s in spans if s[0] <= demo["messages"].index(rich) < s[1]), spans[0])
    chunky = [f"# fb {FB}"] + [m["as_prompt"] for m in demo["messages"][span[0]:span[1]]]
    l4 = "\n".join(chunky)
    who = (raw or {}).get("sender_name", "")
    rx = (rich["reactions"] or [{}])[0]
    media0 = (rich["media"] or [{}])[0]
    # Whichever attachment field the export used — `photos`, `videos`, `audio_files`, `files`,
    # `gifs`, `sticker` — the ladder should show it, or L1 looks like it had no attachment at all.
    raw_uri = ""
    for field in ("photos", "videos", "audio_files", "files", "gifs", "sticker"):
        items = (raw or {}).get(field) or []
        items = items if isinstance(items, list) else [items]
        hit = next((i.get("uri", "") for i in items if isinstance(i, dict) and i.get("uri")), "")
        if hit:
            raw_uri = f"{field}: {hit.split('/')[-1]}"
            break
    cells = []
    # L1 — the export's own object, as a field table so the eye can follow one value across levels
    l1_rows = "".join(
        f"<tr><td>{k}</td><td>{hl(esc(json.dumps(v, ensure_ascii=False)), json.dumps(v, ensure_ascii=False).strip('"'), kind)}</td></tr>"
        for k, v, kind in [
            ("sender_name", who, "who"),
            ("timestamp_ms", (raw or {}).get("timestamp_ms", ""), "time"),
            ("content", (raw or {}).get("content", ""), "text"),
            ("reactions", (raw or {}).get("reactions", []), "react"),
            ("photos[].uri", raw_uri, "att"),
        ] if (k == "content" or v not in ("", None, []))
    )
    l2_rows = "".join(
        f"<tr><td>{k}</td><td>{hl(esc(json.dumps(v, ensure_ascii=False)), json.dumps(v, ensure_ascii=False).strip('"'), kind)}</td></tr>"
        for k, v, kind in [
            ("sender", rich["sender"], "who"),
            ("ts_ms", rich["ts_ms"], "time"),
            ("text", rich.get("text", ""), "text"),
            ("reactions", rich["reactions"], "react"),
            ("media", rich["media"], "att"),
        ] if (k == "text" or v not in ("", None, []))
    )
    l3_html = annotate(esc(l3))
    l4_html = annotate(esc(l4))
    ladder = f"""
      <div class="grid" style="grid-template-columns: repeat(4, 1fr); margin-top:2mm">
        <div class="panel"><h2>L1 · the export's object</h2>
          <table>{l1_rows}</table>
          <p class="note">Names, epoch milliseconds, a typed file reference.</p></div>
        <div class="panel"><h2>L2 · the standardised entry</h2>
          <table>{l2_rows}</table>
          <p class="note">Same facts, named for a consumer: <code>S</code>/<code>O4</code>, a marker per
          reaction, <code>media[]</code> with a kind.</p></div>
        <div class="panel"><h2>L3 · the rendered text</h2>
          <pre style="font-size:10.5px">{l3_html}</pre>
          <p class="note">One string. The date prints once a day, the label is short, and every field
          above became a marker in place.</p></div>
        <div class="panel"><h2>L4 · inside its chunk</h2>
          <pre style="font-size:9.4px">{l4_html[:900]}</pre>
          <p class="note">Chunk <code>#{span[2]}</code>, messages {span[0]}..{span[1]-1}, ~{tokens(l4)} tokens,
          header <code># fb {FB}</code>.</p></div>
      </div>"""

    return f"""
<div class="page">
  <h1>Data → Answers <small>· the corridor and its options · one donor, {vol['conversations']} conversations,
  {vol['messages_K']:.1f}k messages, {vol['audio']['hours']:.1f} h of voice · renderer v3</small></h1>
  {ladder}
  <div class="grid two" style="margin-top:4mm">
    {panel("TL0 · what the browser does before anything leaves the machine", f'''
      <table>
        <tr><th>channel</th><th>conversations</th><th>messages</th><th>text</th><th>text tokens</th><th>prompt tokens</th><th>voice notes</th></tr>
        {rows}
      </table>
      <p class="note">Dropped and counted: {report['dropped_group_threads']:,} conversations with more than two speakers ·
      {report['dropped_conversations_without_subject']} without the donor · {report['dropped_export_lines']:,} lines the export wrote
      about itself · {report['messages_without_content']:,} messages the export left empty. Every name becomes
      <code>S</code> / <code>O&lt;n&gt;</code>; the mapping never leaves the browser.</p>''')}
    {panel("the price of each level, on identical content", f'''
      <table><tr><th>level</th><th>chars</th><th>tokens</th><th>vs L1</th></tr>{grad_rows}</table>
      <p class="note">L1 and L2 are the same bytes — L2 is what a consumer loads; the difference is what it
      <i>names</i> and how it types what it keeps. The ladder from L1 to L4 is {grad[0][2]/grad[3][2]:.1f}× on the same
      words, and every step below L1 is a decision about what to throw away. Colour marks one value travelling down:
      <mark style="background:#ffe9a8">sender</mark>,
      <mark style="background:#cfe6ff">time</mark>,
      <mark style="background:#ffd9ec">reaction</mark>,
      <mark style="background:#d8f2d0">attachment</mark>.</p>''')}
  </div>
</div>"""


def page2(report: dict, threads: list[dict], chunks: list[dict], bundle: Path) -> str:
    """Three consecutive chunks, drawn as blocks, through retrieve → rerank → merge → pull."""
    demo = thread_of(threads, "fb", FB)
    load_bundle, CHUNKERS, ChunkCfg = chunker()
    lib, _, _ = load_bundle(bundle)
    thread = next(t for t in lib if t.platform == "fb" and t.partner == FB)
    fixed = CHUNKERS["fixed"](thread, ChunkCfg(target=512), tokens)[2:5]   # three neighbours, with overlap
    span = (min(c.i0 for c in fixed), max(c.i1 for c in fixed))

    def cell(i: int, tone: str) -> str:
        line = re.sub(r"^\[[^\]]*\]\s*", "", demo["messages"][i]["as_prompt"])
        short = (line[:34] + "…") if len(line) > 34 else (line or "(empty)")
        return f'<div class="m {tone}">{i} · {esc(short)}</div>'

    def strip(lo: int, hi: int, tones: dict[int, str]) -> str:
        return "".join(cell(i, tones.get(i, "")) for i in range(lo, hi))

    # which messages each chunk covers; a message in two chunks gets the overlap tone
    cover = {}
    for c in fixed:
        for i in range(c.i0, c.i1):
            cover.setdefault(i, []).append(c.seq)
    overlap = {i for i, seqs in cover.items() if len(seqs) > 1}
    c1, c2, c3 = fixed
    rows = []
    for c, label in zip(fixed, ("C1", "C2", "C3")):
        tones = {i: ("ov" if i in overlap else "in") for i in range(c.i0, c.i1)}
        rows.append(f'<div class="rowlbl">{label} · messages {c.i0}..{c.i1-1} · {c.n_msgs} msgs · ~{c.n_tokens} tok · id {c.id[:8]}…</div>'
                    f'<div class="strip">{strip(c.i0, c.i1, tones)}</div>')
    shared = sorted(overlap)
    rows.append(f'<div class="rowlbl">stored twice, shown once — messages {shared[0]}..{shared[-1]} '
                f'({len(shared)} of them)</div><div class="strip">' + strip(span[0], span[1], {i: "ov" for i in overlap}) + "</div>")

    table = "".join(
        f"<tr><td>{c['name']}</td><td>{c['chunks']:,}</td><td>{c['tokens']/1e6:.2f}M</td>"
        f"<td>{c['median']}</td><td>{c['min']}</td><td>{c['max']}</td></tr>" for c in chunks)

    return f"""
<div class="page">
  <h1>From rendered messages to a prompt <small>· three neighbouring chunks of fb {FB}, drawn as blocks</small></h1>

  <div class="panel" style="margin-top:3mm">
    <h2>1 · chunking — the same messages, three windows</h2>
    <div class="board">{''.join(rows)}</div>
    <p class="note">Fixed window, target 512, overlap 40%: each chunk repeats the tail of the one before it, which is why
    the shaded messages are stored (and embedded) twice. The day chunker overlaps only where it has to split a long day;
    this strategy overlaps everywhere on purpose — a query that lands on a boundary still finds its region.</p>
  </div>

  <div class="grid two" style="margin-top:4mm">
    <div class="panel"><h2>2 · index — one vector per chunk</h2>
      <div class="board">
        <div class="strip">{[f'<div class="m v">vec C{c.seq}</div>' for c in fixed][0]}</div>
      </div>
      <p class="note">Cosine over every vector, exact, no ANN. The lexical index sits beside it for names, numbers and
      dates, where an embedding blurs what a regex keeps.</p>
    </div>
    <div class="panel"><h2>3 · retrieve → rerank</h2>
      <div class="board">
        <div class="strip"><div class="m hit">C2 · hit</div><div class="m hit">C3 · hit</div><div class="m miss">C1</div></div>
      </div>
      <p class="note">A query returns candidates by score. The reranker (opt-in) scores them as they are — unmerged,
      because that is its native unit — and may reorder them.</p>
    </div>
  </div>

  <div class="grid two" style="margin-top:4mm">
    <div class="panel"><h2>4 · merge — two neighbours become one region</h2>
      <div class="board"><div class="strip"><div class="m merged">C2 ∪ C3 · messages {c2.i0}..{c3.i1-1} · overlap counted once</div></div></div>
      <p class="note">Shown once, cited once. The overlap messages are deduplicated by message index — never by matching
      text, which would collapse two identical-looking messages that really are two.</p>
    </div>
    <div class="panel"><h2>5 · pull — the model asks for more</h2>
      <div class="board"><div class="strip">
        <div class="m ctx">−2</div><div class="m ctx">−1</div><div class="m hit">region</div><div class="m ctx">+1</div><div class="m ctx">+2</div>
      </div></div>
      <p class="note"><code>get(ref, window=2)</code> returns that region plus two messages either side, already rendered.
      Reading around a hit beats searching again, and a repeated message is never shown twice in one run.</p>
    </div>
  </div>

  <div class="panel" style="margin-top:4mm">
    <h2>the three strategies, measured on this corpus</h2>
    <table><tr><th>strategy</th><th>chunks</th><th>indexed tokens</th><th>median msgs</th><th>min</th><th>max</th></tr>{table}</table>
  </div>
</div>"""


POLICY = {
    "call, answered": "kept as a record; the prose is not repeated",
    "call, missed": "kept; direction from the sender",
    "call, event": "kept, flagged as bookkeeping",
    "voice note": "bytes travel; transcript slot empty until STT",
    "photo / video burst": "referenced and counted, bytes stay behind",
    "file · sticker · gif": "referenced, counted, never fetched",
    "shared post": "link + title + owner, nothing fetched",
    "shared profile (no url)": "owner is the whole record",
    "reaction, with time": "when it landed is information the message lacks",
    "reaction, sticker": "kept, its actor with it",
    "deleted by sender": "kept and flagged; counted in report",
    "export left it empty": "kept, marked, counted — never guessed",
    "platform notice": "kept and flagged; duplicates dropped",
    "message deleted (notice)": "kept and flagged",
    "multiline body": "kept literal; never split on",
    "link in the text only": "no marker: already readable",
}

QUIRKS = {
    "facebook": [
        "text arrives double-encoded (`hvornÃ¥r`) and is repaired on the way in",
        "platform lines are written as messages: *connected on Messenger*, *opened this conversation by playing UNO*",
        "`is_unsent` says the sender deleted it; the words are gone, the row is kept",
        "a partner whose account is gone arrives as `Facebook user`, and sometimes as an empty name",
        "calls are prose (“You missed a call from Emma”) plus a duration; some rows are milliseconds",
        "one conversation may be split across 25 export parts and appears twice (DYI + E2EE dump)",
    ],
    "instagram": [
        "same tree as Facebook below the wrapper folder, same reader",
        "calls are event lines (“Audio call ended, 90s”), so direction is often unknowable",
        "a sticker can be the reaction, and arrives as a `.jpg` with an empty emoji",
        "stories and posts share as a link plus the caption Meta captured",
        "`call_duration` is milliseconds more often than Facebook's: an hour is the cut",
    ],
    "whatsapp": [
        "already text: `[DD/MM/YYYY, HH.MM.SS] Sender: text`, three dialects, continuation lines",
        "no reactions and no edits — they are not exported at all",
        "media is named inline: `<attached: 00000025-PHOTO-….jpg>`, and the bytes sit beside the chat",
        "notices arrive as messages with a sender attached: banners, *waiting for this message*, deletions",
        "calls are five shapes (`Missed voice call`, `Voice call. 16 min`, `Silenced voice call. Focus mode`)",
        "one export per conversation, so a donor's ten chats are ten files",
    ],
}


def page3(rows: list[tuple[str, dict[str, str]]]) -> str:
    quirks = "".join(
        f'<div class="panel"><h2>{name}</h2><ul style="margin:0;padding-left:4mm">'
        + "".join(f"<li>{esc(item)}</li>" for item in items) + "</ul></div>"
        for name, items in QUIRKS.items())
    half = (len(rows) + 1) // 2
    def one(chunk):
        out = []
        for label, cells in chunk:
            example = cells["fb"] if cells["fb"] != "—" else next((cells[p] for p in ("ig", "wa") if cells[p] != "—"), "—")
            out.append(f'<tr><td style="width:24%"><b>{esc(label)}</b><br>'
                       f'<span class="note">{esc(POLICY.get(label, ""))}</span></td>'
                       f'<td><code>{esc(example)}</code></td></tr>')
        return "".join(out)
    return f"""
<div class="page">
  <h1>Platforms, and the edge cases <small>· the three exports are not the same file, and the rendered text is where that stops mattering</small></h1>
  <div class="grid" style="grid-template-columns: 1fr 1fr 1fr; margin-top:3mm">{quirks}</div>
  <div class="grid two" style="margin-top:4mm">
    <div class="panel"><h2>edge cases · 1 of 2</h2><table>{one(rows[:half])}</table></div>
    <div class="panel"><h2>edge cases · 2 of 2</h2><table>{one(rows[half:])}</table></div>
  </div>
  <p class="note">One example per case, taken from whichever channel has it, at the level the model reads. `—` only
  appears where no channel in this corpus produced the case: a sticker reaction's five instances all sit in a group
  conversation, which is dropped before any thread is built.</p>
</div>"""


def page4(report: dict, threads: list[dict], bundle: Path) -> str:
    """Two pages, one conversation each, with the fixed window's overlap drawn.

    The day chunker overlaps only where it splits a long day, so its cuts are mostly clean; the fixed
    window is the strategy whose 40% overlap is always there, which is what this sheet is for.
    """
    load_bundle, CHUNKERS, ChunkCfg = chunker()
    lib, _, _ = load_bundle(bundle)
    pages = []
    for platform, partner in (("fb", FB), ("wa", WA)):
        demo = thread_of(threads, platform, partner)
        thread = next(t for t in lib if t.platform == platform and t.partner == partner)
        chunks = CHUNKERS["fixed"](thread, ChunkCfg(target=512), tokens)
        start_at = chunks[1].i0 if len(chunks) > 3 else 0
        cover = {}
        for c in chunks:
            for i in range(c.i0, c.i1):
                cover.setdefault(i, []).append(c)
        blocks = []
        limit = CHAT_MSGS.get((platform, partner), 290)
        for i in range(start_at, min(start_at + limit, len(demo["messages"]))):
            here = cover.get(i, [])
            tone = "ov" if len(here) > 1 else "c0"
            if here and here[0].i0 == i and i > start_at:
                blocks.append(f'<div class="cut"><span>chunk #{here[0].seq} · {here[0].n_msgs} msgs · ~{here[0].n_tokens} tok</span></div>')
            blocks.append(f'<div class="line {tone}">{esc(demo["messages"][i]["as_prompt"])}</div>')
        pages.append(f"""
<div class="page">
  <h1>{platform} · {partner} <small>· {len(demo['messages']):,} messages · {demo['first'][:8]} → {demo['last'][:8]} ·
  fixed window, target 512 tokens, 40% overlap</small></h1>
  <p class="legend"><span class="sw c0"></span> chunk &nbsp;
     <span class="sw ov"></span> overlap — in two chunks, stored and embedded twice, shown once &nbsp;
     <span style="border-top:1.8px dashed #c00;display:inline-block;width:18px;vertical-align:middle"></span> a cut, with the chunk that starts there</p>
  <div class="twocol">{''.join(blocks)}</div>
</div>""")
    return "".join(pages)


def build(bundle: Path, readable: Path, out: Path) -> tuple[Path, Path]:
    out = Path(out).resolve()
    report, threads = load(bundle)
    report["_bundle"] = str(bundle)
    chunks = chunk_stats(bundle)
    edges = edge_table(threads)
    landscape = (f'<!doctype html><meta charset="utf-8"><style>{CSS_LANDSCAPE}</style>'
                 + page1(report, threads, readable) + page2(report, threads, chunks, bundle) + page3(edges))
    portrait = (f'<!doctype html><meta charset="utf-8"><style>{CSS_PORTRAIT}</style>' + page4(report, threads, bundle))
    out.mkdir(parents=True, exist_ok=True)
    land = out / "d2a-corridor-A3-landscape.html"
    port = out / "d2a-chats-A3-portrait.html"
    land.write_text(landscape)
    port.write_text(portrait)
    return land, port


def to_pdf(page: Path, pdf: Path) -> None:
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", page.as_uri()], check=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--readable", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    land, port = build(args.bundle, args.readable, args.out)
    print(f"wrote {land}\nwrote {port}")
    if args.pdf:
        to_pdf(land, args.out / "d2a-corridor-A3-landscape.pdf")
        to_pdf(port, args.out / "d2a-chats-A3-portrait.pdf")
        print("pdfs written")


if __name__ == "__main__":
    main()
