# Preprocess — Messenger export standardiser

Stages: **standardise → merge → dedupe** across 4 export kinds (FB DYI, FB E2EE, Instagram, WhatsApp). Non-UI, Python, runs locally in `preprocess/`.

## Run

```bash
uv run python preprocess/run.py --donor <id> [--with-media]
```

Default = texts, urls, reactions only. `--with-media` also copies media bytes.
Output per donor (groups filtered; only 1:1 survives):

```
out/<donor>/report.json                 per-OTHER stats + source report
out/<donor>/fb/<OTHER>.json             one per non-SUBJECT partner (DYI+E2EE merged)
out/<donor>/fb/<OTHER>__media/          media bytes (only with --with-media)
out/<donor>/ig/...                      same for Instagram, WhatsApp
```

Msg schema: `{ts_iso: YYYYMMDD-HHMMSS (UTC), ts_ms, sender, text, urls[], reactions[{emoji,actor}], media[{name,kind,missing}], system}`.

Media path handling: each `MediaRef` records a concrete locator at parse time —
`{zip: <archive idx>, entry: <zip path>}` (FB DYI) or `{file: <abs path>}` (E2EE dump,
IG tree). Locators survive merge/dedupe; `--with-media` resolves them → copies bytes
into `<OTHER>__media/<name>` (collision-safe), leaving media refs as plain filenames.

## How each platform represents media & content

| | **Facebook DYI** (`messages/…/message_N.json`) | **Facebook E2EE dump** (`<Name>_<id>.json`) | **Instagram** (`messages/…/message_N.json`) | **WhatsApp** (`_chat.txt`) |
|---|---|---|---|---|
| **Schema** | `{participants:[{name}], messages:[{sender_name, timestamp_ms, content, photos[], videos[], audio_files[]}]}` | `{participants:[strings], messages:[{senderName, timestamp, text, type, media[], reactions[], isUnsent}]}` | same as FB DYI | plain text lines |
| **Image** | `photos: [{uri: "…/photos/123.jpg"}]` | `media: [{uri: "./media/<uuid>.jpg"}]` | `photos: [{uri: …}]` | inline `‎<attached: 00000025-PHOTO-….jpg>` |
| **Video** | `videos: [{uri: …}]` | `media: [{uri: "./media/<uuid>.mp4"}]` | `videos: [{uri: …}]` | inline `<attached: 00000044-VIDEO-….mp4>` |
| **Audio / voice** | `audio_files: [{uri: "…/audio/….mp4"}]` (voice notes ship as .mp4) | `media: [{uri: "./media/<uuid>.ogg"}]` | `audio_files: [{uri: …}]` | inline `<attached: 00000024-AUDIO-….opus>` |
| **URL / link** | plain `content` text (no structured field) | `text` plain (no structured field) | plain `content` text | plain line text |
| **Attachment / file** | under `files/` or `content` referring to it | `media: [{uri}]` (flat `media/`) | `content` array w/ `{uri, name?}` | inline `<attached: 00000123-CV 2020.docx>` |
| **Reactions** | `reactions: [{reaction, actor, timestamp}]` | `reactions: [{actor, reaction}]` | `reactions: [{reaction, actor, timestamp}]` | **none** (not exported) |
| **System msg** | `content` boilerplate ("…sent an attachment") + `is_unsent` | `isUnsent` or type flags | `content` boilerplate | plain lines ("Messages and calls are end-to-end encrypted…", "deleted this message", "Security code changed") |
| **Location** | shared object / content URL | `text` w/ link or media | shared object / content URL | plain line `‎Location: https://maps.google.com/…` |
| **Media bytes location** | per-thread `photos/ videos/ audio/` folders in export | flat `media/` folder, shared across all chats | per-thread folders inside `messages/` tree | zip root, `<attached: name>` references |
| **Missing media** | omitted silently | `uri: "Failed to download media"` (flagged `missing`) | omitted silently | `<attached: name>` without matching zip entry (flagged `missing`) |

### WhatsApp text format details

- `[DD/MM/YYYY, HH.MM.SS] Sender: text` — lines may start with `\u200e` (keep regex tolerant)
- continuation lines (no bracket prefix) append to previous message
- `Sender` is display name; the two most-frequent senders = owner + partner (group chats dropped by sender-count)
- timestamps device-local, timezone-naive → parsed as UTC for intra-file ordering; **not** comparable across devices
- media filename embeds type (`PHOTO-`/`VIDEO-`/`AUDIO-`), but kind is inferred from extension too
- no reactions, no read receipts, no structured metadata — text only

## Norms (merge + dedupe)

- **Thread identity**: participant-name signature (`lower|sorted`) joins FB DYI ↔ E2EE dump into one thread; ids `fb001`, `ig001`, `wa001` assigned by message count desc.
- **Dedupe**: `(ts, sender, folded-text)` within a merged thread → drops DYI/E2EE overlap. Count in `report.json` under `dedupe_dropped`.
- **Same person across platforms** stays separate (no cross-platform identity join — deferred, explicit).
- **Groups** (participants > 2) are dropped & counted in `group_threads_dropped`; Instagram/FB labelled by profile names, WhatsApp via sender histogram.
- **Media** is never read — only referenced by `uri` + `kind`. Voice notes keep `kind: audio` even when the file is `.mp4`.

## Known gaps / limitations

- **TODO (heuristics backlog):** thread filter is currently **min-msgs only** (default 20).
  `merge.filter_threads` + `run.py` hold commented-out versions of `min_active_days`
  (distinct calendar days, was 2) and `min_days` (first→last span, was 7) — revisit
  when long-tail definition is settled.

- FB DYI input currently uses **only the 2 fresh zips** in `~/Downloads/...-2026-09-06-*`
  (auto-detected); the full 35-zip Google Drive export would complete the history but
  the sync mount is slow — pass `FB_DYI` directly to `read_classic` for a full set.
- E2EE dump excludes vanish/secret-conversations msgs by design (never in the backup).
- WhatsApp cap: per-chat export truncates to ~10k msgs w/ media (~40k text) — silent tails possible on long chats.
- IG removed E2EE 2026-05-08 → no staleness issue for current exports.