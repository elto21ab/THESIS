# Findings: thesis-deid bundle (2026-09-19) — evidence

**What this file is for:** the measurements behind the upload-side asks, taken by the retrieval
track from the bundle itself, plus the workarounds retrieval carries until they land.
**The action list itself (file:line anchors, acceptance tests, ordering) is not here** — it lives
where the upload session runs: `thesis-deid/docs/BUNDLE_CONTRACT_TODO.md` (items U1–U6).
Edit that file, not this one. If a number here changes, re-run the audit block there.

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

---


## Retrieval-side workarounds (retire these when the U-items land)
| workaround | where | retires with |
|---|---|---|
| rejects >2-sender and `partner=="SUBJECT"` threads at load (18 of 642 here) | `bundle.load_bundle(max_senders=2)` | U3 |
| `donor_id` passed out-of-band (`--donor-id`), stamped into the index manifest, asserted per run, WARN when unset | `cli._resolve_donor_id` / `_donor` | U1 |
| audio rendered as a mask `<attached: AUDIO#k>` instead of its raw filename (which carried the capture stamp) | `bundle._attach_label`, `RENDER_VERSION = v2` | U4 (cosmetic only: names are never rendered) |
| chunk logical-day boundary hardcoded to `Europe/Copenhagen` + `day_start_hour: 5` | `config.yaml:chunk.tz` | U5 |
| nothing reads `sources`, so the name leak is invisible to retrieval — it is a bundle-level PII issue | — | U2 |
| persona is relationship-blind (all partners are `OTHER-nnn`) | `SPEC.md` limitations | U6 |

## U1–U6 index — one line each, details in the work order
| item | ask | anchor (thesis-deid) |
|---|---|---|
| U1 | `report.json.donor_id` from the relay's `<n>`; name the export dir `<donor_id>` | `relay/server.mjs:4-6`, `upload.ts:47`, `bundle.ts:512` |
| U2 | opaque `sources` ids (the WA archive name leaks partner names) — decide U6 first | `parse.ts:134`, `:231`, `:349` |
| U3 | gate **all** group paths on distinct senders (`:167` reads metadata only) + report the drops | `parse.ts:***125***`, ***224***, `:167`; `bundle.ts:512` |
| U4 | drop the capture stamp from media filenames (`let stored = name` keeps the export's own name) | `bundle.ts:371-375` |
| U5 | record `subject_tz` | `bundle.ts` ~:508 |
| U6 | **decision**: may any relationship label be kept? (consent material is local-only in `26_THESIS/legal/`) | `coverage.ts:91-115`, `bundle.ts:297` |
