# Retrieval layer — plan

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
- New uv project `26_THESIS/retrieval/`; artifacts `retrieval/out/<donor>/`.
- MVP scope: fixed + day-heuristic chunkers behind one interface; local Jina embed; index; CLI query; eyeball. **Deferred**: reranker, eval harness, probe/push builder, vLLM client, semantic chunker, MRL.
- Backend: `Embedder` protocol with three impls — `stub` (offline), `jina` (PyTorch/MPS, fidelity reference), `openai` (OpenAI-compatible `/v1/embeddings`, drives **both** local llama.cpp `llama-server` **and** remote vLLM). Full-corpus index is built on the vLLM server; local is for subsets.

## Module layout
```
retrieval/
  io.py        bundle → Thread/Msg dataclasses
  render.py    msg → thesis-deid line (parity w/ minimal.ts)
  chunk/       base.py (protocol) · fixed.py · day_heuristic.py · (semantic.py later)
  chunk_id.py  stable id = hash(platform, partner, ts-span, render-version)
  embed/       base.py (protocol) · local.py (MPS) · (vllm.py later)
  index.py     chunks.jsonl + vectors.npy + manifest.json
  retrieve.py  retrieve(q, filters, k) → hits; span-merge; optional neighbor-expand
  probe.py     STUB push-pack builder
  cli.py       build-index | query
  eval/        (deferred) queries.py · metrics.py
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
embed query w/ Jina query-prompt → cosine top-k → filter (platform/partner/date) → merge overlapping spans → (opt) expand neighbors → return chunks+meta. Reranker deferred.

## Eval (deferred)
Synthetic: hold out msg → its text = query → must retrieve host chunk; `recall@k`/`MRR`/span-precision. ~20 handwritten survey-style probes. Same `retrieve()`, swap chunker → metrics table.

## Deferred / shelved
reranker (jina-v3.5) · eval harness · push/pack probe builder (source TBD: generic vs own surveys) · vLLM client · semantic chunker · MRL dim truncation · cross-platform identity · STT integration.

## MVP run
1. `uv` project. 2. `io`+`render`. 3. `fixed`+`day_heuristic`. 4. download + `embed.local`. 5. `index`. 6. `retrieve`+`cli`. 7. build index on the Downloads bundle; query strings; inspect hits.

## Status (2026-09-19)
- **Done**: io + guards, render (thesis-deid format), `fixed` + `day_heuristic`, `stub`/`jina`/`openai` embedders, index, retrieve (span-merge), cli (`build`/`query`).
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
- Reranker (jina-v3.5) after recall. Eval harness. Probe/push builder. vLLM client config. Jina GGUF local pull.
- Chunker token counter: `openai` backend uses `len//4` (no local tokenizer) → boundaries differ from `jina`; fine for dev, not for comparing strategies across backends.
