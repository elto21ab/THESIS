# Retrieval layer — plan

**Contract spec: `SPEC.md`** (pipeline diagram, tool contract, output shapes, config, TODOs).
This file = decisions + findings + measurements.

RAG over donor chat bundles for persona survey simulation. Corpus ≫ context → chunk + embed + retrieve.

## Input: the bundle (`~/Downloads/donation-2026-09-19`, thesis-deid output)
```
report.json
fb|ig|wa/<OTHER>.json     thread = {id, platform, partner, participants, sources, first, last, media_dir, messages[]}
media/                    audio bytes only; photos/videos/other = refs only
```
msg = `{ts_iso, ts_ms, sender, text, urls[], reactions[], media[{name,kind,missing}], call_s, call_media, call_missed}`.
Bundle schema ≠ legacy `preprocess/` README schema (no `share`/`unsent`/`system` here). Target the bundle.
Scale: 3 channels, 642 threads, 168,294 msgs, ~7.2M prompt-tokens.

### Ordering (correctness, fix first)
Messages within a thread are **not chronological** and the direction varies:
`463 desc / 67 asc / 112 mixed`; 31/642 have >1 directional flip. Cause = concatenated source
dumps (e2ee reverse-chron + DYI/cutover chron). `ts_iso == UTC(ts_ms)` exactly → **always sort by `ts_ms`** before chunking. Slicing the raw array (old BSc) = scrambled chunks.

## Rendering
Reuse thesis-deid `minimal.ts` line format (single source of truth):
`[YYYY-MM-DD HH:MM:SS] LABEL: text <attached: …> <react: …> <call: …> <link: …> <share: …> <unsent>`
- `ts_iso` UTC; label = `SUBJECT`/`OTHER-nnn`.
- MVP: audio stays `<attached: name>` placeholder (stands in for the future transcript). Post-STT = chunk once, then re-embed touched chunks; chunk IDs stable.

## Locked decisions
- Core: `io → sort(ts_ms) → render → chunk → embed → index → retrieve(query,filters)`. One core; callers = agent tool + (later) eval harness.
- Cross-platform separate; channel = filter facet. No identity join.
- Deps live in the **root `pyproject.toml`** (one project for the repo); env = `$UV_PROJECT_ENVIRONMENT` (`~/.uv_main`), never in-tree. Artifacts `retrieval/out/<donor>/`.
- MVP scope: fixed + day-heuristic chunkers behind one interface; local Jina embed; index; CLI query; eyeball. **Deferred**: reranker, eval harness, probe/push builder, vLLM client, semantic chunker, MRL.
- Backend: `Embedder` protocol with three impls — `stub` (offline), `jina` (PyTorch/MPS, fidelity reference), `openai` (OpenAI-compatible `/v1/embeddings`, drives **both** local llama.cpp `llama-server` **and** remote vLLM). Full-corpus index is built on the vLLM server; local is for subsets.

## Module layout (2026-10-13 — 9 files; config loading lives in cli.py)
```
retrieval/            config.yaml (local) · config.server.yaml (point at it with --config)
  bundle.py           bundle → Thread/Msg (+guards) · render_msg · thread_for · logical_day
  chunk.py            ChunkCfg/Chunk · chunk_id (RENDER_VERSION=v2) · fixed + day chunkers
  index.py            Index load/save · lexical blob (lexical.bin + offsets.npy)
  search.py           filters + overlap rule · search_vector · search_lexical · search_any
  models.py           HTTP embeddings + reranking · prompt styles (jina|qwen3|none)
  ops.py              push pack · pull get · payloads · RefRegistry + tool schemas
  cli.py              build | push | pull search|get | smoke   (JSON default, --pretty)
  smoke.py            query → expected pairs, exits 1 on failure
  __main__.py         python -m retrieval
```
## Chunk record
`{id, platform, partner, participants, seq, span:[ts_ms0,ts_ms1], msg_idx:[i,j], n_msgs, n_tokens, day, header, text}`
- `text` = header line (`# fb OTHER-001 2024-01-26 SUBJECT↔OTHER-001`) + rendered lines. Self-containment for embed + model.
- `id` stable across re-embed (STT backfill touches only affected chunks).

## Chunker interface
`chunk_thread(thread, cfg) -> list[Chunk]`; `cfg = {unit, target, min, max, overlap, tz, day_start_hour, both_people}`
- **fixed**: token-budgeted sliding window, boundaries snapped to msg; overlap 40%; both-people soft-expand.
- **day_heuristic** (from notes): per logical day; if content `< min` → attach most-similar neighbor (above/below); if single-party day and still `< min` → repeat; if `min < x < max` merge; else chunk alone.

## Defaults
tokens (jina tokenizer) · target 512 / min 256 / max 1024 · overlap 40% · `tz=Europe/Copenhagen` · `day_start_hour=5` → `logical_date=(local(ts)-5h).date()` · both-people soft · dedup at retrieval · header line on · audio placeholder as text.

## Retrieval
embed query w/ Jina query-prompt → cosine top-k → filter (since/until on logical day) → merge overlapping spans → chunks+meta. Reranker: built, opt-in. `retrieve()` also takes platform/partner for eval/debug; the agent tool does not (see Tool contract).

## Eval (deferred)
Synthetic: hold out msg → its text = query → must retrieve host chunk; `recall@k`/`MRR`/span-precision. ~20 handwritten survey-style probes. Same `retrieve()`, swap chunker → metrics table.

## Deferred / shelved
reranker (jina-v3.5) · eval harness (built; needs real probe queries) · vLLM client · semantic chunker · MRL dim truncation · cross-platform identity · STT integration · query-vector cache (prod HTTP embed ~20 ms, so unneeded for now).

## Env & deps (consolidated 2026-10-06)
One project (root `pyproject.toml`), one env (`$UV_PROJECT_ENVIRONMENT` → `~/.uv_main`). The nested
`retrieval/pyproject.toml` + `uv.lock` + `.venv/` (757 MB) are **deleted** — they violated the
"envs global, never in a work dir" rule. `uv run` from `retrieval/` resolves to the root project and `~/.uv_main`.

```toml
# root pyproject
dependencies = ["numpy>=1.26", "requests>=2.31"]
[project.optional-dependencies]
embed = ["torch>=2.8", "transformers>=4.57", "peft>=0.15.2"]   # local Jina (MPS) only
```
- Base deps = the **prod** path (HTTP to vLLM) + `stub`. `--embedder jina` needs `--extra embed`.
- **ALWAYS `uv sync --inexact`** — plain `uv sync` (or `--extra`) **prunes** unrelated packages from
  the shared `~/.uv_main` (it silently uninstalled patchright/openpyxl/greenlet on 2026-10-06; restored).
  Or install ad-hoc with `uv pip install --python ~/.uv_main …`.
- Single-file scripts: PEP 723 header + `uv run`; `uv lock --script <file>` to pin.
- **No `__pycache__` in the tree**: bytecode is git-ignored (root `.gitignore`), and every entry point runs with
  `python -B` (a cold one-shot process — the compile cost is noise). Or export `PYTHONDONTWRITEBYTECODE=1`
  in the shell and drop the flag everywhere.
- `26_THESIS` now holds **no** in-tree env: `.venv-asr/` (1.1 GB, py3.11, ASR) moved to its home repo `~/Documents/thesis/stt_all/.venv` (2026-10-06), paths/shebangs repaired, imports verified. `stt_all` is not a git repo, so no commit risk.

## MVP run
1. deps in the root `pyproject.toml` (done). 2. `io`+`render`. 3. `fixed`+`day_heuristic`. 4. download + `embed.local`. 5. `index`. 6. `retrieve`+`cli`. 7. build index on the Downloads bundle; query strings; inspect hits.

## Tool contract — push (auto-RAG) vs pull (agentic RAG)
One primitive (`retrieve()`); two entrypoints in `service.py`. Scope = **one donor** (index == donor), so no corpus arg. State (seen-ids, running token budget, turn count) lives in the **caller's loop**, not the service — same fns back an in-process loop and a fresh-process CLI call.

**Agent surface = 2 tools** (fewest knobs; every extra var is one the model can get wrong):
| tool | args | returns |
|---|---|---|
| `search` | `query, k=8, since?, until?` | ranked hits (id, score, day, text…) |
| `get` | `id, window=0` | `window=0` → the chunk; `>0` → ±window raw messages around it |

**push** (offline, not a tool) = `pack(probes, k=4, budget=40_000)`: probes → top-k each → dedupe-by-id → budget-capped. Budget is soft (best hit always kept).

Scratched on purpose: `min_score` (uncalibrated cosine — the model can't pick a threshold), `platform`/`partner` (3 platforms ≈ 0 persona signal; partner ids are opaque pre-discovery), `per_probe` (== `k`), `before`+`after`→`window`, `unit=chunk|msg` (folded into `window`), `max_chars` as a tool arg (transport concern). **Query-time embedder flags are gone**: a call-time model would silently mismatch the index vectors — the manifest is the SSOT and errors loudly if `embedder_cfg` is absent.

CLI = the tool transport (compact JSON, `RETRIEVAL_INDEX` supplies `--index`, `-B` keeps `__pycache__` out of the tree — see Env & deps):
```bash
uv run python -B -m retrieval.cli search "query" -k 8 --json
uv run python -B -m retrieval.cli get    <id> --window 10 --json
uv run python -B -m retrieval.cli pack   --probe "…" -k 4 --budget 40000 --json
```
`--json` is compact (tool result); human rendering is the default. `--max-chars` truncates hit text (the model calls `get` for the rest).

Loop sketch (OpenAI tool-calling, any host): `search` → feed hits → model picks `get`/re-`search`
w/ new args → stop on budget/turns. `search`-only (no prefill) = pure pull; `pack` prefill + live tools = hybrid (recommended default).

---

## Status (2026-09-19)
- **Done**: io + guards, render (thesis-deid format), `fixed` + `day_heuristic`, `stub`/`jina`/`openai` embedders, index, retrieve (span-merge), cli (`build`/`query`).
- **Loop + pairing + citations (2026-10-13)**: `loop.py` (CLI `loop`) = model + tools + state (RefRegistry, shared seen-set, budget/turn caps, `--dry-run` for offline wiring checks); overlapping neighbours are **paired** by chunk count (2→1 item, 3→2, 4→2) with no size cap, so no message is ever presented twice; payloads carry a per-message index limited to *visible* messages (citable `partner:index`, joins back to the bundle). Fixed: "seen" now means *shown* — truncation previously marked unread messages as seen, which made a later `get` return an empty region (388 B instead of 2.8 KB).
- **Overlap = merge, not drop (2026-10-13)**: neighbouring chunks that overlap are unioned into one region (exact message-index dedupe, `2 × chunk.max` cap) in search *and* push; a merged region carries `ids`/`i0..i1`/`n_msgs` and an estimated `n_tokens`. Config is two levels (CLI flag > `config.yaml`) with a complete, validated file and no profiles. `StubEmbedder` → `HashEmbedder` (`embed.backend: hash`, offline vector-path fixture). Bug fixed: message lines are mapped by stored char offsets (`Chunk.line_offsets`) — splitting `text` on newlines misaligned any message containing a newline.
- **S1/S2/S5/S6 pass (2026-10-13)**: config.yaml + profiles replace ~20 flags; CLI is `build | push | pull search|get | smoke` (JSON default, `--pretty`); modules merged 19 → 10 files; prompt styles per model (`jina|qwen3|none`) do the instruction wrapping client-side; audio renders `<attached: AUDIO#k>` and the id drops ts0/ts1 (`RENDER_VERSION=v2`, full corpus rebuilt); `max` is now a hard chunk ceiling (except a single oversized message); `eval` deleted.
- **Lexical index + testing (2026-10-13)**: `--kind lexical` builds a regex-searchable index with **no model** (`lexical.bin` 11.2 MB + `offsets.npy`; full corpus 5469 chunks in **2.3 s** vs ~30 min to embed); `search`/`pack`/`get` dispatch on the index kind through one `search_any`. `smoke` replaces the synthetic eval (6/6 real cases, 0.67 s, offline, non-zero exit on fail). `refs.py` supplies session-local int refs + a constrained `get` schema (`enum`), so the model can never emit a bad chunk id. Full-corpus index at `out/lex-alice` (placeholder donor id).
- **Agent tool layer** (`service.py` + cli `search`/`get`/`pack`, 2026-10-01): push (`pack`) and pull (`search`/`get`) on one `retrieve()`; date filters (`--since/--until`); compact `--json` payloads; `get(id, window)` returns the chunk or ± raw msgs; embedder read from the manifest (`embedder_cfg`) — no call-time override. Knob audit trimmed the tool to 2 fns / 6 args. Stateless — the loop owns seen-ids/budget. `RETRIEVAL_INDEX` fixes the donor per run. **Remaining**: probe source for push (survey instrument, not generic) + the loop itself (caller-side).
- **Uploader bugs fixed** in thesis-deid (see `FINDINGS-uploader.md`): (1) sort messages by ts in `finaliseThread`; (2) group guard on distinct senders, not participant metadata. 26/26 tests, typecheck ok.
- **Chunker bug fixed**: overlap was implemented by rewinding the segmentation pointer, which re-triggered the same day/max boundary → cascades of tiny duplicate chunks (min 8 tok, 368 sub-min in one thread). Now segmentation advances non-overlapping and overlap is only a text prefix.
- **Validated**: Jina v5 small (PyTorch/MPS) cross-lingual EN↔DA; OpenAI client against local `llama-server` (bge-m3) cross-lingual; real target chunk retrieves (rank ~3, beaten by lexical neighbours → reranker's job).

### Embedding speed (why vLLM builds the full index)
| path | ~ms/chunk | full corpus (5469 chunks) |
|---|---|---|
| stub (no model) | ~0 | ~3 s |
| Jina v5 small, PyTorch MPS bf16 | ~1000 @1k tok | ~40+ min |
| Jina v5 small, PyTorch MPS fp16 | ~854 @1k tok | ~30 min |
| bge-m3 Q8, llama.cpp Metal (via server) | ~290 | ~25 min |
Batch plateaus (MPS saturated); cost scales with seq length. Conclusion: local = subsets only; full build on vLLM.

### Local dev backend (llama.cpp) — context sizing gotcha (2026-10-13)
`-c` is the **total** context, divided across `-np` slots: `-c 4096` (the old recipe) gave ~1024
tok/slot with 4 slots, so any chunk above that made the server cancel the task and return **HTTP 500**
mid-build — and a build only writes at the end, so ~6 min of embedding was lost. Working recipe for
the full corpus (max chunk ≈ 2047 tok):

```bash
llama-server -m ~/.models/embed/bge-m3-q8_0.gguf --embedding --port 8091 -c 16384 -ub 4096 -np 4
```
TODO: shard/checkpoint the embed step so a long build survives a failure.

### Local dev backend (llama.cpp)
```bash
llama-server -m ~/.models/embed/bge-m3-q8_0.gguf --embedding --port 8091 -c 4096 -ub 2048
uv run python -m retrieval.cli build --out out/dev --chunker day \
  --embedder openai --model bge-m3 --no-task-prompt --sample 200
```
Same-model local dev: pull `jinaai/jina-embeddings-v5-text-small-retrieval-GGUF` and drop `--no-task-prompt`
(the `-retrieval` GGUF bakes the task adapter; keep the `Query: `/`Document: ` prefixes).

## Engine/stack decisions (2026-09-26)
- **Prod** = vLLM (CUDA): `jina-embeddings-v5-text-small` + `jina-reranker-v3.5`. Server needs vLLM >= 0.30 (0.30 resolves `JinaForRanking`; verified on Mac CPU).
- **Local dev** = **vllm-metal** serving `Qwen3-Embedding-0.6B-8bit` + `Qwen3-Reranker-0.6B-mlx-8Bit` over the OpenAI-compatible API. One client (`OpenAIEmbedder`/`HttpReranker`) for local and prod; only the served model differs. Locked for **library/API parity**.
- **Keep** (not deleted, for other projects): brew `llama.cpp`, the fork at `~/builds/llama.cpp-jina` (branch `qwen3-swa`), all Jina GGUF/weights in `~/.models/embed/`, the `~/.venv-vllm-metal` env.

### Measured (Mac; not like-for-like — different models/quants/batching)
| task | model | engine | speed |
|---|---|---|---|
| embed | Qwen3-Emb-0.6B (8-bit) | vllm-metal | 423 ms/item |
| embed | Jina v5 (F16) | llama.cpp | 430 ms/item |
| embed | Jina v5 (bf16) | transformers/MPS | 1184 ms/item |
| rerank | Jina v3.5 | llama.cpp fork | 405 ms/doc |
| rerank | Jina v3.5 | vLLM-CPU | 3450 ms/doc |
| rerank | Qwen3-Rer-0.6B | vllm-metal | 13 ms/doc |

### Backlog (post-MVP)
- **Recency-weighted scoring**: inflate scores of newer chunks (time-decay curve TBD) — a function over
  `Chunk.ts1`/day combined with cosine. Discuss once MVP is stable. Recorded 2026-09-19.

### Open / next
- **Push probe source**: fix the probe set (survey instrument questions, not generic) → drives `pack` defaults.
- **Caller loop**: OpenAI tool-calling over the CLI (`search`/`get`), seen-set + budget + turn cap. Pure pull (no prefill) vs hybrid (`pack` prefill + live tools).
- **Full-corpus index**: dev index built locally with llama.cpp + `bge-m3` Q8 (`out/vec-bge`, config
  kept out of tree at build time) while the UCloud GPU path is arranged; prod index = `jina-embeddings-v5-text-small`
  on UCloud vLLM (needs a live job + SSH tunnel). Local vLLM embeddings: **vllm-metal does serve
  Qwen3-Embedding** via `pooling/backends/decoder/models/qwen3.py` (decoder pooling; the `encoder/models/`
  dir I checked first only covers bge-m3/xlm-roberta — do not conclude support from one directory listing
  again). Served id must match exactly: `mlx-community/Qwen3-Embedding-0.6B-8bit` (~45 s startup).
- Chunker token counter: `openai` backend uses `len//4` (no local tokenizer) → boundaries differ from `jina`; fine for dev, not for comparing strategies across backends.
- `get --window>0` reloads the bundle per call (~0.5 s, scans `bundle/<platform>/*.json`); cache partner→path if the loop hammers it.
