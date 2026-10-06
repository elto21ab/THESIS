# Findings: thesis-deid bundle (2026-09-19) — three uploader bugs

Found while building the retrieval layer. All three change what the corpus actually
contains, so they matter beyond chunking. `bundle = ~/Downloads/donation-2026-09-19`.

## 1. Messages are never sorted (chronology broken)
`src/core/standard/format.ts:133 finaliseThread` sets `first`/`last` but never sorts
`t.messages`. Order is whatever `parse.ts` pushed: files iterated **path-sorted** (Meta's
`message_1.json` = newest → descending), then `merge.ts` appends source B after source A.

Measured: **463 desc / 67 asc / 112 mixed** of 642 threads; 31 threads have >1 directional
flip. Example `fb/OTHER-001.json`: one flip at msg 8418 — e2ee dump (desc) then DYI/cutover (asc).
`ts_iso == UTC(ts_ms)` exactly (0/16149 off).

Fix: stable sort by `ts` in `finaliseThread` (before dedupe + write).
Impact: any consumer that trusts array order (including the old BSc chunker) scrambles context.

## 2. Group threads leak through
Group guard reads the export's `participants` **metadata**, not actual senders:
`parse.ts:117 basket.names.size > 2`, `parse.ts:159 participants.length > 2`. When that
metadata is missing/truncated, groups pass. The WhatsApp reader (`:323`, counts real senders)
is unaffected.

Measured: **18/642 threads** carry 3–34 distinct senders.
- `fb/SUBJECT.json` — worst: 34 senders, `participants:["SUBJECT"]`, `partner:"SUBJECT"`;
  a Facebook group ("Vigtige beskeder", rename/join events). `finalise` then picks
  `participants[0]` as partner → a fake 1:1.
- `fb/OTHER-016.json` (8 senders), `fb/OTHER-074/054/011` (4), plus 13 with 3.

Fix: gate on **distinct senders**, or reconcile against `participants`; drop + count.
Interim: the retrieval loader rejects `>2 senders` or `partner=="SUBJECT"` (18 rejects).

## 3. PII survives in leaked-group system lines
Those group threads keep real names in message text, e.g.
`fb/SUBJECT.json`: "Elias Salvador Smidt Torjani, Robert Krogh Groth and others joined the chat".
Name replacement covers sender labels, not free text / structured system notices. Group
join/rename events are structured and trivially scrubable; free-text names are the hard case
(NER). Flagging because it is object-level PII in a "de-identified" bundle.

## Not bugs, but relevant
- `preprocess/` (Python) schema ≠ bundle schema (no `share`/`unsent`/`system` fields here).
  Target the bundle.
- Audio is the only media with bytes; photos/videos/other are references only.
