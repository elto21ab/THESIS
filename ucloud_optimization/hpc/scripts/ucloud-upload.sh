#!/usr/bin/env bash
# Setup a dedicated UCloud upload key; upload one local directory via resumable rsync.
set -Eeuo pipefail
IFS=$'\n\t'

KEY="${UCLOUD_KEY:-$HOME/.ssh/ucloud_upload}"
HOST="${UCLOUD_HOST:-ssh.cloud.sdu.dk}"
USER="${UCLOUD_USER:-ucloud}"
DEST_ROOT="${UCLOUD_DEST:-/work/llama-inference/data}"

usage() {
  cat <<'EOF'
Usage:
  ucloud-upload.sh setup [name]
  ucloud-upload.sh upload PORT [SOURCE_DIR] [UPLOAD_NAME]

Examples:
  ./ucloud-upload.sh setup airidas
  ./ucloud-upload.sh upload 2386 "$HOME/Desktop/data" airidas

Env overrides: UCLOUD_KEY, UCLOUD_HOST, UCLOUD_USER, UCLOUD_DEST
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing '$1'"; }
valid_name() { [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]; }

setup_key() {
  local name="${1:-$(id -un)}"
  need ssh-keygen
  mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
  if [[ ! -f "$KEY" ]]; then
    printf 'Creating dedicated key: %s\nUse a strong passphrase.\n' "$KEY"
    ssh-keygen -t ed25519 -a 64 -C "ucloud-upload:${name}" -f "$KEY"
  elif [[ ! -f "$KEY.pub" ]]; then
    ssh-keygen -y -f "$KEY" > "$KEY.pub"
  fi
  chmod 600 "$KEY"; chmod 644 "$KEY.pub"

  if [[ "$(uname -s)" == Darwin ]]; then
    ssh-add --apple-use-keychain "$KEY" || true
    command -v pbcopy >/dev/null && pbcopy < "$KEY.pub"
    printf 'Public key copied to clipboard.\n'
  fi
  printf '\nSend ONLY this public key to the UCloud owner:\n'
  cat "$KEY.pub"
  printf '\nNever send %s or its passphrase.\n' "$KEY"
}

upload() {
  local port="${1:-}" source="${2:-$HOME/Desktop/data}" name="${3:-$(id -un)}"
  [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1 && port <= 65535)) || die "Invalid PORT"
  valid_name "$name" || die "UPLOAD_NAME allows letters, numbers, dot, underscore, hyphen"
  [[ -d "$source" ]] || die "Source dir missing: $source"
  [[ -f "$KEY" ]] || die "Key missing. Run: $0 setup $name"
  need ssh; need rsync

  local remote="$DEST_ROOT/$name"
  local ssh_cmd="ssh -i $KEY -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p $port"
  printf 'Uploading %s/ → %s@%s:%s/\n' "${source%/}" "$USER" "$HOST" "$remote"

  # Host-key default remains strict/interactive: never silently trust a new host.
  ssh -i "$KEY" -o IdentitiesOnly=yes -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 -p "$port" "$USER@$HOST" \
    "mkdir -p -- '$remote'"
  rsync -a --partial --info=progress2 -e "$ssh_cmd" \
    "${source%/}/" "$USER@$HOST:$remote/"

  local local_count local_bytes remote_stats
  local_count="$(find "$source" -type f | wc -l | tr -d ' ')"
  local_bytes="$(du -sk "$source" | awk '{print $1}')"
  remote_stats="$(ssh -i "$KEY" -o IdentitiesOnly=yes -p "$port" "$USER@$HOST" \
    "printf '%s ' \"\$(find '$remote' -type f | wc -l)\"; du -sk '$remote' | awk '{print \$1}'")"
  printf 'Done. Local files/KiB: %s/%s; remote files/KiB: %s\n' \
    "$local_count" "$local_bytes" "$remote_stats"
}

case "${1:-}" in
  setup) shift; setup_key "${1:-}" ;;
  upload) shift; upload "$@" ;;
  -h|--help|help|'') usage ;;
  *) usage >&2; exit 2 ;;
esac
