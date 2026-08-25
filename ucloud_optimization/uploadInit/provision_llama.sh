#!/usr/bin/env bash
# Fetch + build llama.cpp; fetch a GGUF model. Idempotent; no sudo/package-manager mutations.
set -Eeuo pipefail
IFS=$'\n\t'

LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/minicpm5-1b}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}" # Pin tag/commit in prod.
MODEL_REPO="${MODEL_REPO:-openbmb/MiniCPM5-1B-GGUF}"
MODEL_FILE="${MODEL_FILE:-MiniCPM5-1B-Q4_K_M.gguf}"
MODEL_REVISION="${MODEL_REVISION:-main}" # Pin commit in prod.
MODEL_SHA256="${MODEL_SHA256:-}"         # Strongly recommended in prod.
BACKEND="${BACKEND:-cpu}"                # cpu | cuda | metal
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-}"

log() { printf '[provision] %s\n' "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing '$1'"; }

for cmd in git cmake curl; do need "$cmd"; done
if [[ -z "$JOBS" ]]; then
  JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 2)"
fi

case "$BACKEND" in
  cpu)   backend_args=(-DGGML_NATIVE=OFF) ;;
  cuda)  need nvcc; backend_args=(-DGGML_CUDA=ON -DGGML_NATIVE=OFF) ;;
  metal) backend_args=(-DGGML_METAL=ON -DGGML_NATIVE=OFF) ;;
  *) die "BACKEND must be cpu, cuda, or metal" ;;
esac

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  [[ ! -e "$LLAMA_DIR" ]] || die "$LLAMA_DIR exists but is not a git repo"
  log "Cloning llama.cpp → $LLAMA_DIR"
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

log "Checking out llama.cpp ref: $LLAMA_CPP_REF"
git -C "$LLAMA_DIR" fetch --tags origin "$LLAMA_CPP_REF"
git -C "$LLAMA_DIR" checkout --detach FETCH_HEAD
LLAMA_COMMIT="$(git -C "$LLAMA_DIR" rev-parse HEAD)"

log "Building llama.cpp ($BACKEND, $JOBS jobs)"
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DLLAMA_CURL=ON \
  "${backend_args[@]}"
cmake --build "$LLAMA_DIR/build" --config "$BUILD_TYPE" -j "$JOBS"

mkdir -p "$MODEL_DIR"
MODEL_PATH="$MODEL_DIR/$MODEL_FILE"
MODEL_URL="https://huggingface.co/$MODEL_REPO/resolve/$MODEL_REVISION/$MODEL_FILE?download=true"

if [[ ! -s "$MODEL_PATH" ]]; then
  log "Downloading $MODEL_REPO/$MODEL_FILE → $MODEL_PATH"
  # .part + rename prevents consumers seeing a partial model; -C - resumes interruptions.
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "$MODEL_PATH.part" "$MODEL_URL"
  mv "$MODEL_PATH.part" "$MODEL_PATH"
else
  log "Model already present: $MODEL_PATH"
fi

if [[ -n "$MODEL_SHA256" ]]; then
  need shasum
  printf '%s  %s\n' "$MODEL_SHA256" "$MODEL_PATH" | shasum -a 256 -c -
fi

SERVER="$LLAMA_DIR/build/bin/llama-server"
[[ -x "$SERVER" ]] || die "Build finished but $SERVER is missing"

cat <<EOF
Ready.
llama.cpp commit: $LLAMA_COMMIT
model:             $MODEL_PATH
server:            $SERVER

Run:
  "$SERVER" -m "$MODEL_PATH" --host 127.0.0.1 --port 8080
EOF
