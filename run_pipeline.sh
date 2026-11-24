#!/usr/bin/env bash
# One donation, end to end, on the machine that will hold it.
#
#   ./run_pipeline.sh --participant p999-g7olrgxx99w18qum [--stt] [--embed] [--root DIR]
#
# What it does, in order, writing only to persistent places:
#   1. pull the newest bundle from the donation drive   → $ROOT/data/<participant>/
#   2. merge the survey files (newest send per instrument) → surveys.json
#   3. render + chunk (days)                              → 3_rendered/, 4_chunks/
#   4. optional: speech to text                           → transcripts.json + re-render
#   5. optional: build the retrieval index                → index/
#   6. write logs/run-<timestamp>.log and results/<participant>/manifest.json
#
# The token is never written down here. It comes from the environment, and the two ways to get it there
# both avoid shell history:
#
#   read -rsp 'UCloud API token: ' UCLOUD_TOKEN; echo; export UCLOUD_TOKEN
#   # or, locally, straight out of Bitwarden without printing it:
#   # aac run --domain cloud.sdu.dk --env UCLOUD_TOKEN=password -- ./run_pipeline.sh …
#
# The drive root is not "/" — a workspace-scoped token is authorised for its own drive path and answers
# 403 on anything else. /12418519 is the donation drive as of 2026-11.
set -euo pipefail

PARTICIPANT=""
ROOT="${THESIS:-$HOME/thesis}"
DRIVE_ROOT="${UCLOUD_DRIVE_ROOT:-/12418519}"
DO_STT=0
DO_EMBED=0
CHUNKER="${CHUNKER:-day}"

while [ $# -gt 0 ]; do
  case "$1" in
    --participant) PARTICIPANT="${2:?}"; shift 2 ;;
    --root)        ROOT="${2:?}"; shift 2 ;;
    --drive-root)  DRIVE_ROOT="${2:?}"; shift 2 ;;
    --stt)         DO_STT=1; shift ;;
    --embed)       DO_EMBED=1; shift ;;
    --chunker)     CHUNKER="${2:?}"; shift 2 ;;
    -h|--help)     sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$PARTICIPANT" ] || { echo "--participant is required" >&2; exit 2; }
[ -n "${UCLOUD_TOKEN:-}" ] || {
  echo "UCLOUD_TOKEN is not set. It is deliberately not read from a file in this repo:" >&2
  echo "  read -rsp 'UCloud API token: ' UCLOUD_TOKEN; echo; export UCLOUD_TOKEN" >&2
  exit 2
}

HERE="$(cd "$(dirname "$0")" && pwd)"
DATA="$ROOT/data/$PARTICIPANT"
LOGS="$ROOT/logs"
RESULTS="$ROOT/results/$PARTICIPANT"
STAMP="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
LOG="$LOGS/run-$PARTICIPANT-$STAMP.log"

mkdir -p "$DATA" "$LOGS" "$RESULTS" "$ROOT/models/hf" "$ROOT/index"
export HF_HOME="${HF_HOME:-$ROOT/models/hf}"
export TMPDIR="${TMPDIR:-/work/tmp}"; mkdir -p "$TMPDIR"
export UCLOUD_DRIVE_ROOT="$DRIVE_ROOT"

run() {  # run <label> <command…> — every step is logged with its wall time
  local label="$1"; shift
  local start; start=$(date +%s)
  printf '\n== %s\n' "$label" | tee -a "$LOG"
  if "$@" 2>&1 | tee -a "$LOG"; then
    printf -- '-- %s ok (%ss)\n' "$label" "$(( $(date +%s) - start ))" | tee -a "$LOG"
  else
    printf -- '-- %s FAILED (%ss)\n' "$label" "$(( $(date +%s) - start ))" | tee -a "$LOG"
    exit 1
  fi
}

cd "$HERE"

run "pull $PARTICIPANT" \
  uv run python -B pull_donation.py --participant "$PARTICIPANT" --out "$DATA" --surveys

# `--readable` is deliberately absent: it needs the raw export tree, which a server never receives.
run "views + chunks ($CHUNKER)" \
  uv run python -B make_views.py --bundle "$DATA" --out "$DATA" --chunks "$CHUNKER"

if [ "$DO_STT" = 1 ]; then
  run "speech to text" uv run python -B -m prep stt --bundle "$DATA" --out "$RESULTS/transcripts.json"
  run "attach transcripts" uv run python -B -m prep transcripts --bundle "$DATA" --transcripts "$RESULTS/transcripts.json"
  run "re-render" uv run python -B -m prep render --bundle "$DATA"
  run "parity" uv run python -B -m prep parity --bundle "$DATA"
fi

if [ "$DO_EMBED" = 1 ]; then
  run "index build" uv run python -B -m retrieval build --bundle "$DATA" --out "$ROOT/index/$PARTICIPANT"
fi

# What came out, so a later run can tell whether anything changed without re-reading the corpus.
python3 - "$DATA" "$RESULTS/manifest.json" "$STAMP" "$CHUNKER" <<'PY' | tee -a "$LOG"
import json, sys, os, glob
data, target, stamp, chunker = sys.argv[1:5]
report = json.load(open(os.path.join(data, "report.json")))
manifest = {
    "ran_at": stamp,
    "participant": report.get("donor_id"),
    "render_version": report.get("render_version"),
    "conversations": (report.get("volume") or {}).get("conversations"),
    "prompt_tokens_M": ((report.get("volume") or {}).get("prompt") or {}).get("tokens_M"),
    "chunker": chunker,
    "views": sorted(os.path.basename(p) for p in glob.glob(os.path.join(data, "*")) if os.path.isdir(p)),
    "chunks": sum(1 for _ in glob.glob(os.path.join(data, "4_chunks", "*", "*.md"))),
}
json.dump(manifest, open(target, "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY

printf '\nlog: %s\nmanifest: %s\n' "$LOG" "$RESULTS/manifest.json"
