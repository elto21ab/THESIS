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


def corpus_gradient(threads: list[dict]) -> list[tuple[str, int, int]]:
    """chars and tokens for the whole corpus at each level, so the ladder has a price tag."""
    json_chars = sum(len(json.dumps(m, ensure_ascii=False)) for t in threads for m in t["messages"])
    l3 = sum(len(m["as_prompt"]) for t in threads for m in t["messages"])
    l4 = sum(len(re.sub(r"^\[[^\]]*\]\s*", "", m["as_prompt"])) for t in threads for m in t["messages"])
    json_tokens = sum(tokens(json.dumps(m, ensure_ascii=False)) for t in threads for m in t["messages"])
    return [
        ("L1 · raw export message", json_chars, json_tokens),
        ("L2 · standardised entry", json_chars, json_tokens),
        ("L3 · rendered text", l3, tokens("x" * 0) + sum(tokens(m["as_prompt"]) for t in threads for m in t["messages"])),
        ("L4 · lean text, no stamp", l4, sum(tokens(re.sub(r"^\[[^\]]*\]\s*", "", m["as_prompt"])) for t in threads for m in t["messages"])),
    ]


# ── chunking ─────────────────────────────────────────────────────────────────
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
.page { break-after: page; }
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
.sw { display: inline-block; width: 9px; height: 9px; border-radius: 1mm; vertical-align: middle; }
.line { font-family: "SF Mono", Menlo, monospace; font-size: 10px; white-space: pre-wrap; padding: 0.5mm 1.2mm; }
.c0 { background: #e6efff; }
.c1 { background: #ffeccf; }
.ov { background: #ffd6d6; }
.cut { border-top: 1.8px dashed #c00; margin: 1mm 0 0.6mm; position: relative; }
.cut span { background: #c00; color: #fff; font-size: 8px; padding: 0 1.2mm; border-radius: 0 0 1.5mm 1.5mm; }
"""


def panel(title: str, body: str, cls: str = "") -> str:
    return f'<div class="panel {cls}"><h2>{title}</h2>{body}</div>'


def page1(report: dict, threads: list[dict], readable: Path) -> str:
    vol = report["volume"]
    rows = "".join(
        f"<tr><td>{p}</td><td>{d['conversations']}</td><td>{d['messages_K']:.1f}k</td>"
        f"<td>{d['text']['chars_M']:.2f}M</td><td>{d['text']['tokens_M']:.2f}M</td>"
        f"<td>{d['prompt']['tokens_M']:.2f}M</td><td>{d['audio']['files']}</td></tr>"
        for p, d in report["platforms"].items()
    )
    grad = corpus_gradient(threads)
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
    ladder = f"""
      <table>
        <tr><th style="width:15%">level</th><th>what it is</th><th style="width:44%">this message</th></tr>
        <tr><td><b>L1 · raw</b></td><td>the export's own JSON object, one file per part</td>
            <td><pre>{esc(l1)[:1400]}</pre></td></tr>
        <tr><td><b>L2 · entry</b></td><td>the standardised message: labels instead of names, markers as fields</td>
            <td><pre>{esc(l2)[:1400]}</pre></td></tr>
        <tr><td><b>L3 · text</b></td><td>the one string the model reads (<code>as_prompt</code>)</td>
            <td><pre>{esc(l3)}</pre></td></tr>
        <tr><td><b>L4 · chunk</b></td><td>L3 lines under a header, with the span it covers
            (<code>#{span[2]}</code>, messages {span[0]}..{span[1]-1}, ~{tokens(l4)} tokens)</td>
            <td><pre>{esc(l4)[:1400]}</pre></td></tr>
      </table>"""

    return f"""
<div class="page">
  <h1>Data → Answers <small>· the corridor and its options · one donor, {vol['conversations']} conversations,
  {vol['messages_K']:.1f}k messages, {vol['audio']['hours']:.1f} h of voice · renderer v3</small></h1>
  <div class="grid two" style="margin-top:3mm">
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
      <p class="note">L1 and L2 are the same JSON object — L2 is what a consumer loads; the difference is what it
      <i>names</i>. The ladder from L1 to L4 is {grad[0][2]/grad[3][2]:.1f}× on the same words. That factor is the
      whole argument for the register question: every level below L1 is a decision about what to throw away.</p>''')}
  </div>
  <div class="grid" style="margin-top:4mm">
    {panel("one message, four levels — a call with a reaction on it", ladder, "wide")}
  </div>
</div>"""


def page2(report: dict, threads: list[dict], chunks: list[dict], bundle: Path) -> str:
    flow = """
    <div class="flow">
      <div class="step"><b>thread</b><br>one conversation, chronological<br><span class="k">629–649 of them</span></div>
      <div class="sep">→</div>
      <div class="step"><b>chunk</b><br>day group or token window, 40% overlap<br><span class="k">~9k chunks</span></div>
      <div class="sep">→</div>
      <div class="step"><b>embed</b><br>one vector per chunk, cosine<br><span class="k">exact, no ANN</span></div>
      <div class="sep">→</div>
      <div class="step"><b>retrieve</b><br>top-k per query or per probe<br><span class="k">vector + lexical</span></div>
      <div class="sep">→</div>
      <div class="step"><b>rerank</b><br>cross-encoder on unmerged chunks<br><span class="k">opt-in</span></div>
      <div class="sep">→</div>
      <div class="step"><b>merge</b><br>overlapping neighbours → one item<br><span class="k">never twice</span></div>
      <div class="sep">→</div>
      <div class="step"><b>prompt</b><br>header + lines, under the glossary<br><span class="k">the only text the model sees</span></div>
    </div>"""
    rows = "".join(
        f"<tr><td>{c['name']}</td><td>{c['chunks']:,}</td><td>{c['tokens']/1e6:.2f}M</td>"
        f"<td>{c['median']}</td><td>{c['min']}</td><td>{c['max']}</td></tr>" for c in chunks)

    # where the cuts fall in the FB conversation
    spans = spans_of(bundle, "fb", FB)[:8]
    demo = thread_of(threads, "fb", FB)
    marks = "".join(
        f'<tr><td>#{s[2]}</td><td>{s[0]}..{s[1]-1}</td><td>{s[1]-s[0]}</td>'
        f'<td>~{sum(tokens(m["as_prompt"]) for m in demo["messages"][s[0]:s[1]])}</td></tr>' for s in spans)

    return f"""
<div class="page">
  <h1>Chunking and retrieval <small>· the same thread, from text to a prompt</small></h1>
  <div class="grid" style="margin-top:3mm; grid-template-columns: 1fr;">
    {panel("the corridor", flow)}
  </div>
  <div class="grid two" style="margin-top:4mm">
    {panel("chunking strategies, on this corpus", f'''
      <table><tr><th>strategy</th><th>chunks</th><th>indexed tokens</th><th>median msgs</th><th>min</th><th>max</th></tr>{rows}</table>
      <p class="note"><b>Day</b>: one logical day per chunk (05:00 → 05:00 local), a day over <code>max</code> is split by the
      fixed window. <b>Fixed</b>: a token window that snaps boundaries to messages. Overlap is 40% of <code>target</code>:
      neighbours repeat a third of their messages, which is why indexed tokens exceed the prompt total — and why
      deduplication has to exist at all.</p>''')}
    {panel(f"where the cuts fall — fb {FB}, first chunks", f'''
      <table><tr><th>chunk</th><th>messages</th><th>msgs</th><th>~tokens</th></tr>{marks}</table>
      <p class="note">Ids are <code>blake2b(version|platform|partner|i0|i1)</code>: span-addressed, so a transcript arriving
      later re-embeds the same span instead of orphaning it. The model never sees them — it cites <code>ref</code> integers,
      which the tool resolves.</p>''')}
  </div>
  <div class="grid two" style="margin-top:4mm">
    {panel("what deduplication means here", '''
      <p class="note">Two neighbouring chunks that both rank are <b>paired into one item</b>, not shown twice and not dropped:
      the union span is what the model reads, and the repeated messages are counted once by message index, never by text.
      Merging happens <i>after</i> reranking — the cross-encoder judges unmerged chunks, which is its native unit, and
      merging then frees slots so the fill continues down the ranked list.</p>''')}
    {panel("what the model is given", '''
      <p class="note">Header (<code># fb O4</code>) and the rendered lines. Nothing else: no JSON, no ids, no metadata it cannot
      read. The glossary above the chunks explains <code>S</code>, <code>O4</code> and every marker once, so a line never
      spends tokens explaining itself.</p>''')}
  </div>
</div>"""


def page3(rows: list[tuple[str, dict[str, str]]]) -> str:
    body = "".join(
        f"<tr><td><b>{esc(label)}</b></td>" + "".join(
            f'<td class="{"ok" if cells[p] != "—" else "no"}">' + (f"<code>{esc(cells[p][:150])}</code>" if cells[p] != "—" else "—") + "</td>"
            for p in ("fb", "ig", "wa")) + "</tr>"
        for label, cells in rows)
    return f"""
<div class="page">
  <h1>Edge cases <small>· one real example per case, at the rendered level · <code>—</code> means the platform cannot produce it</small></h1>
  <div class="grid" style="margin-top:3mm; grid-template-columns: 1fr">
    {panel("", f'''
      <table>
        <tr><th style="width:17%">case</th><th style="width:28%">facebook</th><th style="width:28%">instagram</th><th style="width:27%">whatsapp</th></tr>
        {body}
      </table>''')}
  </div>
</div>"""


def page4(report: dict, threads: list[dict], bundle: Path) -> str:
    cols = []
    for platform, partner in (("fb", FB), ("wa", WA)):
        demo = thread_of(threads, platform, partner)
        spans = spans_of(bundle, platform, partner)
        # the stretch with the most chunks, capped so it fits a page
        start, end = spans[0][0], min(spans[0][0] + 62, len(demo["messages"]))
        if len(spans) > 6:
            start = spans[1][0]
            end = min(start + 62, len(demo["messages"]))
        blocks = []
        for i in range(start, end):
            chunk_ids = [s for s in spans if s[0] <= i < s[1]]
            cls = "ov" if len(chunk_ids) > 1 else f"c{chunk_ids[0][2] % 2 if chunk_ids else 0}"
            cut = next((s for s in spans if s[0] == i), None)
            if cut and i > start:
                blocks.append(f'<div class="cut"><span>cut · chunk #{cut[2]} starts here</span></div>')
            blocks.append(f'<div class="line {cls}">{esc(demo["messages"][i]["as_prompt"])}</div>')
        cols.append(panel(
            f"{platform} · {partner} · {len(demo['messages']):,} messages · {demo['first'][:8]} → {demo['last'][:8]}",
            "".join(blocks)))
    return f"""
<div class="page">
  <h1>Two conversations, as the study chunks them
    <small>· shaded bands are chunks, red bands are the 40% overlap where two chunks share messages, dashed lines are cuts</small></h1>
  <p class="legend"><span class="sw c0"></span> chunk even &nbsp;
     <span class="sw c1"></span> chunk odd &nbsp;
     <span class="sw ov"></span> overlap — shown twice in the index, once to the model &nbsp;
     <span style="border-top:1.6px dashed #c00;display:inline-block;width:18px;vertical-align:middle"></span> a cut</p>
  <div class="cols">{''.join(cols)}</div>
</div>"""


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
