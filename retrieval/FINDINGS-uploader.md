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

---

# Upload-session TODO (raised by retrieval, 2026-10-13)
These belong to the uploader / standard-format layer, **not** to retrieval. Retrieval currently
works around them (guards, `--donor-id`, filename masks) and should be able to drop those
workarounds once the bundle carries the right metadata.

### U1. Donor id must be in the bundle
Today: folder = `donation-2026-09-19` (date only), `report.json` says only `donor:"SUBJECT"`.
Nothing distinguishes two donors' bundles; retrieval has to be told the id out-of-band
(`--donor-id`) and stamps it into the index manifest.
**Want:** folder named `<donor_id>/` and/or `donor_id` (the registry id donors are given) written
into `report.json`. Retrieval then reads it and asserts it per run.

### U2. `sources` leaks real display names
39 of 42 distinct `sources` values are original export filenames containing partner names:
`"wa:WA/WhatsApp Chat - Aksel.zip"`, `"… - Emma ❤️.zip"`, `"… - Alisa (Rebecca).zip"`.
Same class as bug 3 (object-level PII in a bundle described as de-identified).
**Want:** opaque ids (`wa:src-17`) or drop the field. Note the trade-off: `sources` is currently
the only relationship hint that survives labelling (see U6).

### U3. Move the data-quality gates to upload
`max_senders == 2` (distinct senders, not `participants` metadata) and `partner != "SUBJECT"`
are re-implemented defensively in retrieval. They are upload/bundle-quality properties.
**Want:** upload drops those threads and reports them in `report.json` (count + reason).

### U4. Media filenames embed the capture timestamp (redundant + leak)
Scheme: `NNNNNNNN-KIND-YYYY-MM-DD-HH-MM-SS.ext`, e.g.
`00000008-AUDIO-2023-02-20-13-33-33.opus`. The timestamp is *identical* to the containing
message's `ts_ms` (verified: `ts_ms=1676900013000` ≡ `2023-02-20 13:33:33Z`), so it duplicates
data we already have and leaks the capture time into anything that renders the filename.
**Want:** opaque media ids (`<thread>/<seq>` or a hash) with the kind kept as a field, not in the
name. The message already carries the timestamp; nothing is lost.
Why it matters here: retrieval renders `<attached: KIND>` masks for photos/videos but currently
must pass audio's raw filename through (it is the future transcript slot). Opaque ids would let
the mask stay stable *and* the transcript join stay exact.

### U5. Subject timezone
The logical-day chunk boundary needs one tz (UTC is stored; only the boundary is local). A
travelling subject makes a fixed value slightly wrong. Recording the subject's tz (or per-source
tz) at upload lets retrieval default to it instead of hardcoding `Europe/Copenhagen`.

### U6. Partner identity / relationship hints
After labelling, `SUBJECT`/`OTHER-nnn` carry no relationship information. The only surviving
signal was the export filename (U2), which is the thing being scrubbed. Decide what may be kept
(e.g. an explicit, consented relationship label per partner) — it materially affects how well the
persona can imitate how the donor talks to a given person.
