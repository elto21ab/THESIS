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

# Upload-side work order — bundle contract fixes

**Written for a session with no prior context.** Every symptom below is reproducible from the
bundle alone with the commands given; nothing here depends on reading the retrieval code first.

## 0. Orientation

**0.1 Artifacts.** `bundle = ~/Downloads/donation-2026-09-19` — one donor, the *contract* that the
rest of the pipeline consumes. Consumer: `/Users/e/Documents/thesis/_active/26_THESIS/retrieval`
(`SPEC.md` = contract, `PLAN.md` = decisions). Uploader: **`/Users/e/Documents/thesis/_active/thesis-deid`**
(TypeScript, `src/core/standard/{format,parse,merge,bundle,coverage}.ts`, relay in `relay/server.mjs`).

> **⚠ Its checkout is `donation-ucloud`, 67 commits behind its remote** (`master` 89 behind) — pull
> before changing anything; the §1 bug fixes are newer than what is on disk.

> **Actionable version of U1–U6 (file:line anchors + acceptance tests + ordering) lives where the
> upload session runs: `thesis-deid/docs/BUNDLE_CONTRACT_TODO.md`. That file is the single source of
> truth for the work; this one keeps the measurements and the consumer's assumptions.**

**0.2 What the consumer already depends on — do not break these.**
- `report.json`: `donor`, `generated_at`, `platforms`, `totals` are read and stamped into the index
  manifest (provenance + donor guard).
- Thread files `fb|ig|wa/<OTHER>.json`: `messages[]` with `{ts_ms, sender, text, urls[],
  reactions[], media[{name, kind, missing}], call_s, call_media, call_missed}`; order is assumed
  **ascending `ts_ms`** (see §1.1).
- `media/<name>`: the file name is referenced by `media[].name` but is **never rendered into a
  prompt** — the consumer renders kind masks (`<attached: PHOTO>`) and, for audio, `<attached:
  AUDIO#k>` where `k` is the index inside that message's `media[]`. So renaming files is safe;
  *changing the array order or dropping entries is not*.
- The bundle path is stored in the index manifest and re-read later for `get --window`.

**0.3 Status of §1's bugs: both are fixed in `thesis-deid` after this bundle was generated**
(stable sort by `ts_ms` in `finaliseThread`; group guard on distinct senders). The bundle on disk
**predates** those fixes, so the consumer guards defensively and rejects 18 threads. Re-export
before the real experiment.

## 1. Symptom audit — run this first (~30 s)

```bash
cd ~/Downloads/donation-2026-09-19
jq -r '[.donor, (.donor_id // "ABSENT")]|@tsv' report.json        # U1 → SUBJECT<TAB>ABSENT
basename "$PWD"                                                   # U1 → donation-2026-09-19 (no id)
jq -r '.sources[]?' */*.json | sort -u > /tmp/src.txt
wc -l < /tmp/src.txt; grep -c 'Chat - ' /tmp/src.txt               # U2 → 42 ; 35
for f in */*.json; do [ "$(basename $f)" = report.json ] && continue
  n=$(jq -r '[.messages[].sender]|unique|length' "$f"); [ "$n" -gt 2 ] && echo "$f $n"; done | wc -l   # U3 → 18
ls media | grep -c AUDIO; ls media | wc -l                         # U4 → 270 ; 891
jq -r '.messages[]|select((.media//[])[]?.kind=="audio")|"\(.ts_ms) \(.media[0].name)"' wa/OTHER-847.json | head -1
#   → 1676900013000 00000008-AUDIO-2023-02-20-13-33-33.opus   (timestamp duplicated in the name)
jq -r '[.messages[]|keys[]]|unique|join(",")' wa/OTHER-847.json | grep -ci 'tz\|offset'   # U5 → 0
```
Observed 2026-10-13: as annotated. If any number differs, the bundle has changed — re-read that item.

## U1. The bundle carries no donor id  ⟶ blocks per-donor identity
**Symptom.** Folder is `donation-2026-09-19` (a date) and `report.json` has only
`donor: "SUBJECT"`; there is no `donor_id`. Two donors' bundles are indistinguishable, and the
consumer must be told the id out-of-band (`retrieval build --donor-id`, then `RETRIEVAL_DONOR` /
`--expect-donor` asserts it per run). If it is unset the build prints a warning and the manifest
records `donor_id: null`.
**Where it comes from.** The donor is issued an id at donation time (the README's
`donate.torjani.app/?<donor-id>` flow) — that id is never carried into the export.
**Want.** `report.json.donor_id` (preferred) **and** the bundle directory named `<donor_id>`.
**Acceptance.**
`jq -r .donor_id report.json` → non-empty; then
`cd retrieval && uv run python -B -m retrieval build --bundle <bundle> --out /tmp/idx --kind lexical`
prints `donor : <donor_id>` with no WARN, and
`RETRIEVAL_DONOR=<donor_id> uv run python -B -m retrieval pull search "x" --index /tmp/idx` succeeds
while a wrong value aborts with `donor mismatch`.
**Coupling.** Consumer resolution order: `--donor-id` → `report.json.donor_id|subject_id` → bundle
dir name (unless it matches `donation-*`).

## U2. `sources` leaks real display names  ⟶ PII in a "de-identified" bundle
**Symptom.** 42 distinct `sources` values; **35** contain a partner's display name, e.g.
`"wa:WA/WhatsApp Chat - Aksel.zip"`, `"… - Emma ❤️.zip"`, `"… - Alisa (Rebecca).zip"` (39 match a
broader name-ish pattern incl. `_`). Same class as the older §3 finding: object-level PII.
**Where.** The standard-format layer records which export file each thread came from.
**Want.** Opaque source ids (`wa:src-17`) — or drop the field. **Trade-off:** `sources` is currently
the *only* surviving relationship hint (see U6), so decide U6 first.
**Acceptance.** `jq -r '.sources[]?' */*.json | sort -u | grep -c 'Chat - '` → `0`, and there is not
even a `;` — but retrieval renders nothing from `sources`, so nothing downstream breaks.
**Coupling.** none (retrieval never reads `sources`).

## U3. Data-quality gates belong to upload, not to every consumer
**Symptom.** 18/642 threads carry 3–34 distinct senders (worst: `fb/SUBJECT.json`, 34 senders,
`participants: ["SUBJECT"]`, `partner: "SUBJECT"` → a fake 1:1). Also
`dropped_conversations_without_subject` exists in `report.json`, so the concept is already there.
The consumer re-implements the gates (rejects `>2` distinct senders or `partner == "SUBJECT"`).
**Want.** Upload drops those threads and reports them: `report.json.dropped_threads` with
`{count, reason, files[]}` (or extend the existing `dropped_conversations_without_subject`).
**Acceptance.** the audit loop in §1 prints nothing for a fresh bundle, and `report.json` names the
dropped files.
**Coupling.** once present, retrieval's `bundle.max_senders` guard can go away.

## U4. Media filenames embed the capture timestamp  ⟶ duplicated data + leak
**Symptom.** `media/` names are `NNNNNNNN-KIND-YYYY-MM-DD-HH-MM-SS.ext`
(`00000008-AUDIO-2023-02-20-13-33-33.opus`, 270 AUDIO of 891 files). The embedded time is
**identical** to the containing message's `ts_ms` (`1676900013000` ≡ `2023-02-20 13:33:33Z`,
verified in §1) — so it duplicates data we already have and leaks capture time to anything that
shows a filename.
**Where.** Whatever layer writes the sanitised media dir (grep `thesis-deid` for `AUDIO` / the
`KIND-` pattern).
**Want.** Opaque names (`<thread-id>/<seq>` or a hash) with `kind` staying a *field*
(`media[].kind`), not part of the name.
**Acceptance.** `ls media | grep -c AUDIO` → 0 for the new scheme; every `media[].name` still
resolves to an existing file in `media/`; a build + `pull search` still renders `<attached: AUDIO#k>`.
**Coupling.** safe by construction: the consumer never renders the name, and chunk ids derive from
`(platform, partner, i0, i1)` — **a rename does not invalidate any index**.
Note the current workaround: audio used to render its raw filename into prompts (<45 chars/line,
date leak); the consumer now masks it (`RENDER_VERSION = v2`).

## U5. Subject timezone is not recorded (day boundaries are a guess)
**Symptom.** No timezone anywhere: messages carry only `ts_ms`/`ts_iso` (UTC) and no `tz`/`offset`
key; `report.json` has no tz either. The consumer groups chunks by *logical day* with a fixed
`chunk.tz: Europe/Copenhagen` + `day_start_hour: 5`, so a travelling subject gets slightly wrong
day boundaries (timestamps themselves stay UTC — nothing is lost, only boundaries shift).
**Want.** `report.json.subject_tz` (e.g. `Europe/Copenhagen`) and, if cheap, a per-source tz.
**Acceptance.** the key exists and the consumer defaults to it instead of hardcoding.
**Coupling.** `chunk.tz` in `retrieval/config.yaml`; changing tz re-buckets chunks → re-embed
(chunk ids include the span, so ids are stable, but boundaries differ).

## U6. Partner identity / relationship hints — **a decision, not a task**
**Symptom.** After labelling, `SUBJECT` / `OTHER-nnn` carry no relationship information, yet how the
donor talks depends heavily on *who* they are talking to. The only surviving hint is the export
filename (U2), i.e. exactly the thing being scrubbed. `report.json.platforms.<p>` does carry
`subject_confirmed_by_donor`, and `legal/` (local only) holds the consent/DPIA material.
**Decide (with the supervisor / donor consent):** may any relationship label be kept — e.g. a coarse
category, or a consented label per partner? If not, accept that the persona is relationship-blind
and say so in the thesis limitations.
**Acceptance.** a written decision in this file + (if kept) a field in the thread file.

## 2. Ordering
1. **U1** and **U4** (contract changes, mechanical), then **U3** (quality gate reuses `report.json`
   plumbing).
2. **U2** only after **U6** is decided — they trade off directly.
3. **U5** whenever.
4. Re-export the bundle after U1–U4 (§1's audit should come back clean), then re-run the consumer:
   `uv run python -B -m retrieval build --kind lexical --donor-id <id> --out out/<id> --bundle <new>`
   and `uv run python -B -m retrieval smoke` (expect 6/6).
