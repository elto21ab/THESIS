#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""A3 sheets for the supervisor workshop: the data→answer options, printed.

Everything on them is measured on this donor's corpus, not illustrative. Four panels:

  1  the pipeline as it exists, with the counts at each stage
  2  the rendering gradient: the same messages at four levels of leanness, and what each costs
  3  chunking and retrieval: the knobs, and what they do to this corpus
  4  candidate conversations, by platform, for the live walk-through

    uv run 26_THESIS/make_workshop.py --bundle "26_THESIS/data/2_donation upload-2026-10-27" \
        --out "26_THESIS/data/6_workshop" [--pdf]
"""
from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'


def iso(ms: int | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime(fmt)


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


# ── levels of leanness ───────────────────────────────────────────────────────
def level1(m: dict) -> str:
    """The file itself: every field, named. What a consumer reads before any rendering."""
    keys = []
    keys.append(f'"ts": {m["ts_ms"]}')
    keys.append(f'"sender": "{m["sender"]}"')
    keys.append(f'"text": {json.dumps(m.get("text", ""))}')
    if m.get("reactions"):
        keys.append('"reactions": ' + json.dumps(m["reactions"], ensure_ascii=False))
    if m.get("media"):
        keys.append('"media": ' + json.dumps([{k: r[k] for k in ("kind",) } for r in m["media"]], ensure_ascii=False))
    if "call_s" in m:
        keys.append(f'"call": {{"s": {m["call_s"]}, "missed": {json.dumps(m.get("call_missed", False))}}}')
    return "{" + ", ".join(keys) + "}"


def level2(m: dict) -> str:
    """v2: absolute stamp and full label on every message."""
    prompt = m["as_prompt"]
    return re.sub(r"^\[[^\]]*\] ([SO]\d*|[A-Z]+-\d+):", lambda g: f"[{iso(m['ts_ms'], '%Y-%m-%d %H:%M:%S')}] {m['sender']}:", prompt)


def level3(m: dict) -> str:
    """v3: the date once a day, short labels."""
    return m["as_prompt"]


def level4(m: dict) -> str:
    """Lean: no stamp inside the line at all; the day lives in the chunk header."""
    return re.sub(r"^\[[^\]]*\]\s*", "", m["as_prompt"])


def tokens(text: str) -> int:
    """The tool's own estimator (`estimateTokens` in `minimal.ts`), ported so every figure on the
    sheet agrees with `report.json`: letters cost 1 token per 3.9 characters, digits 1 per 1.6,
    every standalone symbol 1.3."""
    if not text:
        return 0
    import math
    # ASCII classes, because JavaScript's \w is ASCII: a Danish letter is a symbol there, and the
    # point of porting this was to agree with the figure in report.json to the token.
    letters = sum(math.ceil(len(m) / 3.9) for m in re.findall(r"[A-Za-z_]+", text))
    digits = sum(math.ceil(len(m) / 1.6) for m in re.findall(r"\d+", text))
    symbols = re.findall(r"[^A-Za-z0-9_\s]", text)
    return letters + digits + (math.ceil(len(symbols) * 1.3) if symbols else 0)


def gradient(threads: list[dict]) -> dict[str, dict[str, float]]:
    out = {}
    for name, fn in (("L1 JSON", level1), ("L2 v2 lines", level2), ("L3 v3 lines", level3), ("L4 lean", level4)):
        chars = 0
        ntok = 0
        msgs = 0
        for t in threads:
            for m in t["messages"]:
                s = fn(m)
                chars += len(s)
                ntok += tokens(s)
                msgs += 1
        out[name] = {"chars": chars, "tokens": ntok, "per_msg": chars / max(1, msgs)}
    return out


def chunk_stats(bundle: Path) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).parent / "retrieval"))
    from retrieval.bundle import load_bundle
    from retrieval.chunk import CHUNKERS, ChunkCfg

    threads, rejects, _ = load_bundle(bundle)
    count = tokens
    rows = []
    for name, cfg in (("day", ChunkCfg()), ("fixed · 512/40%", ChunkCfg(target=512)), ("fixed · 256/40%", ChunkCfg(target=256))):
        chunker = CHUNKERS["day"] if name == "day" else CHUNKERS["fixed"]
        chunks = [c for t in threads for c in chunker(t, cfg, count)]
        if not chunks:
            continue
        sizes = sorted(c.n_msgs for c in chunks)
        rows.append({
            "name": name,
            "chunks": len(chunks),
            "tokens": sum(c.n_tokens for c in chunks),
            "median": sizes[len(sizes) // 2],
            "min": sizes[0],
            "max": sizes[-1],
        })
    return rows


def pick_examples(threads: list[dict]) -> dict[str, dict]:
    """One conversation per platform: the richest in edge cases that is written in English."""
    EN = re.compile(r"\b(the|you|and|is|to|of|what|that|for|have|just|with|about|like|can|do|not)\b", re.I)
    DA = re.compile(r"\b(jeg|og|er|det|du|ikke|til|at|en|et|har|vi|s\u00e5|mig|dig|hvad|kan|skal|om|med|der|den|de|v\u00e6re|lige|ogs\u00e5)\b", re.I)
    wanted = ["call", "voice", "photo", "react", "share", "unsent", "empty", "notice", "multiline", "link"]
    best: dict[str, tuple[int, float, dict]] = {}
    for t in threads:
        msgs = t["messages"]
        text = " ".join(m.get("text") or "" for m in msgs)
        en, da = len(EN.findall(text)), len(DA.findall(text))
        english = en / max(1, en + da)
        has = {
            "call": any("call_s" in m for m in msgs),
            "voice": any(r["kind"] == "audio" for m in msgs for r in m["media"]),
            "photo": any(r["kind"] == "photos" for m in msgs for r in m["media"]),
            "react": any(m["reactions"] for m in msgs),
            "share": any("share" in m for m in msgs),
            "unsent": any(m.get("unsent") for m in msgs),
            "empty": any("<no content>" in m["as_prompt"] for m in msgs),
            "notice": any("system" in m for m in msgs),
            "multiline": any("\n" in (m.get("text") or "") for m in msgs),
            "link": any(m["urls"] for m in msgs),
        }
        score = sum(has.values())
        platform = t["platform"]
        if english < 0.25:
            continue
        if platform not in best or score > best[platform][0]:
            best[platform] = (score, english, t)
    return {p: v[2] for p, v in best.items()}


def excerpt(thread: dict, limit: int = 16) -> list[str]:
    """A stretch that shows the greatest *variety* of edge cases, not the most of one.

    Scoring by count picked a burst of eleven photos in a row and nothing else; the point of the
    excerpt is that a reader sees one call, one reaction, one empty message, one share and a voice
    note inside a page. Distinct marker kinds first, repeated-attachment bursts penalised, both
    speakers preferred.
    """
    kinds = ("<voice", "<react:", "<call:", "<share:", "<link:", "<unsent>", "<no content>", "<attached: FILE",
             "<attached: STICKER", "<attached: GIF", "<attached: VIDEO", "<attached: PHOTO")
    msgs = thread["messages"]
    best_at, best = 0, -1.0
    for i in range(0, max(1, len(msgs) - limit)):
        window = msgs[i:i + limit]
        lines = [m["as_prompt"] for m in window]
        distinct = len({k for line in lines for k in kinds if k in line})
        repeats = sum(line.count("<attached: PHOTO") for line in lines)
        voices = len({m["sender"] for m in window})
        score = distinct * 10 - repeats * 0.5 + voices
        if score > best:
            best_at, best = i, score
    return [m["as_prompt"] for m in msgs[best_at:best_at + limit]]


def esc(s: str) -> str:
    return html.escape(s)


CSS = """
@page { size: A3 landscape; margin: 10mm; }
* { box-sizing: border-box; }
body { font: 12px/1.35 "Helvetica Neue", Arial, sans-serif; color: #111; margin: 0; }
h1 { font-size: 22px; margin: 0 0 2mm; }
h2 { font-size: 14px; margin: 0 0 1.5mm; text-transform: uppercase; letter-spacing: .04em; color: #444; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6mm; }
.panel { border: 1px solid #bbb; border-radius: 3mm; padding: 4mm; break-inside: avoid; }
.wide { grid-column: 1 / -1; }
table { border-collapse: collapse; width: 100%; font-size: 11px; }
th, td { text-align: left; padding: 1.2mm 2mm; border-bottom: 1px solid #ddd; vertical-align: top; }
th { font-weight: 600; color: #333; }
code, pre { font-family: "SF Mono", Menlo, monospace; font-size: 10.5px; }
pre { background: #f5f5f5; padding: 2mm 3mm; border-radius: 2mm; white-space: pre-wrap; margin: 0; }
.note { color: #555; font-size: 11px; }
.k { display: inline-block; background: #eee; border-radius: 2mm; padding: 0 1.5mm; font-size: 10px; }
.big { font-size: 26px; font-weight: 700; }
.dim { color: #777; }
.page { break-after: page; }
"""


def build_html(report: dict, threads: list[dict], grad: dict, chunks: list[dict], picks: dict[str, dict]) -> str:
    vol = report["volume"]
    plat = report["platforms"]
    rows = "".join(
        f"<tr><td>{p}</td><td>{d['conversations']}</td><td>{d['messages_K']:.1f}k</td>"
        f"<td>{d['text']['tokens_M']:.2f}M</td><td>{d['prompt']['tokens_M']:.2f}M</td>"
        f"<td>{d['audio']['files']}</td></tr>"
        for p, d in plat.items()
    )
    grad_rows = "".join(
        f"<tr><td>{k}</td><td>{v['chars']/1e6:.2f}M</td><td>{v['tokens']/1e6:.2f}M</td>"
        f"<td>{v['per_msg']:.0f}</td><td>{v['tokens']/grad['L1 JSON']['tokens']*100:.0f}%</td></tr>"
        for k, v in grad.items()
    )
    chunk_rows = "".join(
        f"<tr><td>{c['name']}</td><td>{c['chunks']:,}</td><td>{c['tokens']/1e6:.2f}M</td>"
        f"<td>{c['median']}</td><td>{c['min']}</td><td>{c['max']}</td></tr>"
        for c in chunks
    )

    # one message, four levels
    # One message carrying as much as a message can: a reaction, an attachment and a call or a
    # share, so the four levels differ by more than punctuation.
    def richness(m: dict) -> int:
        line = m["as_prompt"]
        return sum(k in line for k in ("<react:", "<attached:", "<voice", "<call:", "<share:", "<link", "<unsent>"))
    rich = max((m for t in threads for m in t["messages"]), key=richness)
    levels = "".join(
        f"<tr><td>{name}</td><td><code>{esc(fn(rich))}</code></td><td>{len(fn(rich))} ch</td></tr>"
        for name, fn in (("L1 JSON", level1), ("L2 v2", level2), ("L3 v3", level3), ("L4 lean", level4))
    )

    cards = []
    for p in ("fb", "ig", "wa"):
        t = picks.get(p)
        if not t:
            continue
        marks = sorted({k for line in excerpt(t) for k in
                        ["<attached: PHOTO", "<attached: VIDEO", "<attached: FILE", "<attached: GIF", "<attached: STICKER",
                         "<voice", "<react:", "<call:", "<share:", "<link:", "<unsent>", "<no content>"]
                        if any(k in l for l in excerpt(t))})
        body = "\n".join(excerpt(t))
        cards.append(f"""
        <div class="panel">
          <h2>{p} · {t['partner']} · {len(t['messages']):,} messages · {t['first'][:8]} → {t['last'][:8]}</h2>
          <p class="note">markers present: {' '.join(f'<span class="k">{esc(m)}</span>' for m in marks)}</p>
          <pre>{esc(body)}</pre>
        </div>""")

    return f"""<!doctype html><meta charset="utf-8"><style>{CSS}</style>
<div class="page">
  <h1>Data → Answers: the pipeline, and the options inside it</h1>
  <p class="note">Every figure is measured on this donor's corpus ({vol['conversations']} conversations,
  {vol['messages_K']:.1f}k messages, {vol['audio']['hours']:.1f} h of voice notes). Renderer v3.</p>
  <div class="grid">
    <div class="panel">
      <h2>TL0 · upload — what the browser does before anything leaves the machine</h2>
      <table>
        <tr><th>channel</th><th>conversations</th><th>messages</th><th>text tokens</th><th>prompt tokens</th><th>voice notes</th></tr>
        {rows}
      </table>
      <p class="note">Dropped, and counted in <code>report.json</code>: {report['dropped_group_threads']:,} conversations with more
      than two speakers, {report['dropped_conversations_without_subject']} without the donor, {report['dropped_export_lines']} lines the
      export wrote about itself, {report['messages_without_content']:,} messages the export left empty. Every name inside becomes
      <code>S</code> / <code>O&lt;n&gt;</code>; the mapping never leaves the browser.</p>
    </div>
    <div class="panel">
      <h2>TL1 · the rendering gradient — tokens are the currency</h2>
      <table><tr><th>level</th><th>chars</th><th>tokens</th><th>chars / message</th><th>vs L1</th></tr>{grad_rows}</table>
      <p class="note">L1 is the file itself, L2 is the old line format (absolute stamp, full label on every message),
      L3 is what the tool writes now (date once a day, <code>S</code>/<code>O1</code>), L4 drops the in-line stamp entirely
      and relies on the chunk header. The gap between L1 and L4 is a factor of
      {grad['L1 JSON']['tokens']/grad['L4 lean']['tokens']:.1f}× on identical content.</p>
    </div>
    <div class="panel wide">
      <h2>the same message at each level</h2>
      <table><tr><th>level</th><th>rendered</th><th>size</th></tr>{levels}</table>
    </div>
  </div>
</div>

<div class="page">
  <h1>TL1 · chunking and retrieval — the knobs, and what they cost</h1>
  <div class="grid">
    <div class="panel">
      <h2>chunking</h2>
      <table><tr><th>strategy</th><th>chunks</th><th>indexed tokens</th><th>median msgs</th><th>min</th><th>max</th></tr>{chunk_rows}</table>
      <p class="note"><b>Day</b> keeps one logical day per chunk (a day runs 05:00 → 05:00 local) and splits a day that
      exceeds <code>max</code>. <b>Fixed</b> slides a token window and snaps boundaries to messages. Overlap is 40% of
      <code>target</code>: consecutive chunks repeat a third of their messages, which is why indexed tokens exceed the
      corpus' prompt tokens.</p>
    </div>
    <div class="panel">
      <h2>retrieval</h2>
      <table>
        <tr><th>decision</th><th>what we do</th><th>why</th></tr>
        <tr><td>candidates</td><td>exact cosine over ~4k chunks, no ANN</td><td>at ~9k chunks a matmul is faster than any index</td></tr>
        <tr><td>lexical</td><td>regex scan beside the vectors</td><td>names, numbers and dates survive where embeddings blur them</td></tr>
        <tr><td>overlap</td><td>merge after reranking</td><td>the cross-encoder judges unmerged chunks; merging then frees slots and never shows a message twice</td></tr>
        <tr><td>reranker</td><td>built, opt-in</td><td>costs a model call per candidate; measure before enabling</td></tr>
        <tr><td>mode</td><td>push (probes known) / pull (model asks)</td><td>a survey has its questions in advance; an open interview does not</td></tr>
        <tr><td>window</td><td><code>get(ref, window=k)</code> → ±k messages</td><td>reading around a hit beats searching again</td></tr>
      </table>
    </div>
  </div>
  <h2 style="margin-top:6mm">TL2 · answer — the model never sees the corpus, only retrieved text</h2>
  <div class="panel wide">
    <p class="note">Retrieved chunks are placed in the prompt under a glossary (<code>S</code> = the donor, <code>O3</code> = the
    third person they were talking to, <code>fb/ig/wa</code> = the channel, what every marker means). The model answers as
    the donor would. Two open questions worth the supervisor's time: <b>which register</b> the rendered text should use
    (plain chat, LEAN, or a controlled language such as ASD-STE100), and <b>whether the profile/condense layer</b> sits
    before retrieval (summary in the prompt) or after (the model reads only raw chunks).</p>
  </div>
</div>

<div class="page">
  <h1>Candidate conversations for the live walk-through</h1>
  <p class="note">One per channel, chosen for English content and edge-case coverage: calls, voice notes, photos,
  reactions, shared posts, unsent messages, empty messages, multiline bodies, links.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    report, threads = load(args.bundle)
    grad = gradient(threads)
    chunks = chunk_stats(args.bundle)
    picks = pick_examples(threads)
    args.out.mkdir(parents=True, exist_ok=True)
    page = args.out / "d2a-options.html"
    page.write_text(build_html(report, threads, grad, chunks, picks))
    print(f"wrote {page}")
    for name, value in grad.items():
        print(f"  {name:14s} {value['chars']/1e6:6.2f}M chars  {value['tokens']/1e6:5.2f}M tokens")
    for row in chunks:
        print(f"  {row['name']:16s} {row['chunks']:5d} chunks  {row['tokens']/1e6:5.2f}M tokens")
    for p, t in picks.items():
        print(f"  pick {p}: {t['partner']} · {len(t['messages'])} messages")
    if args.pdf:
        pdf = args.out / "d2a-options.pdf"
        subprocess.run([
            CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf}", page.as_uri(),
        ], check=True, capture_output=True)
        print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
