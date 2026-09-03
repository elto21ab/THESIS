# UCloud CLI workflow — comparison + reproducible setup

**Verdict: [`rlrs/ucloud-cli`](https://github.com/rlrs/ucloud-cli) = primary. [CBS-HPC/ucloud-api](https://github.com/CBS-HPC/ucloud-api) = skip for this project.** Neither replaces in-box persistence rules (setsid/tmux) — CLI gets you *to* the box; [vllm/runbook/PERSISTENCE_RULES.md](vllm/runbook/PERSISTENCE_RULES.md) still governs *on* it.

## Comparison

| | rlrs/ucloud-cli | CBS-HPC/ucloud-api |
|---|---|---|
| Auth | interactive login (user/pass + MFA), token refresh, Firefox import | static `UCLOUD_TOKEN` API token in `.env` |
| Submit | `--from <known-good-job-id>` (= UI "Import parameters"), dry-run default, `--execute` to fire; overrides: `--product-id --time --mount --app-version` | template job via `UCLOUD_TEMPLATE_JOB_ID`; `--mount` flags self-declared "experimental and unverified" |
| SSH | `jobs ssh <id>` / `--print-only` → resolves **dynamic port automatically** (kills #1 UI-scraping pain) | probes SSH endpoint, bounded noninteractive commands |
| Stop | `jobs stop <id>` (kills hold-to-confirm UI hack) | `jobs stop` |
| Status | `jobs status <id> --poll` + `jobs browse` w/ remaining time | `jobs wait` (blocks till running, prints SSH cmd) |
| Extras | files/drives ls, wallets, per-user GPU-h accounting, raw `request` escape hatch | python-job upload/run/download pipeline, utilization analysis, delivery zip |
| Fit for us | interactive GPU jobs, custom mounts/toolchain, vLLM serving | batch python extract→deliver; CBS-internal 0.1.x RC |
| Friction | `uv tool install`, login once | `uv sync` + `.env` + mint API token in UI + maintain template job |

CBS dealbreakers: experimental mounts (our jobs hinge on `/work/LCPP_OffloadTesting` attach), batch-job orientation (no fit for long-lived vLLM server + ad-hoc bench), extra token to rotate. Its `jobs wait` idea is nice; rlrs `status --poll` covers it.

## Setup (you + coworker, ~5 min)

```bash
# 1. install (no clone needed)
uv tool install "git+https://github.com/rlrs/ucloud-cli.git"

# 2. login once (MFA prompt); refresh later w/ `ucloud auth refresh`
ucloud auth login --username <sdu-username>
ucloud projects use --project <project-id>   # or pick interactively

# 3. SSH key (unchanged — CLI shells out to ssh, key still required)
#    UCloud → Resources → SSH keys → add pubkey; per Mac session:
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

## Daily loop (replaces UI + patchright driving)

```bash
JID=$(ucloud jobs submit --from <known-good-job-id> --time 04:00 --execute --output json | jq -r '.id')
ucloud jobs status $JID --poll                      # wait for RUNNING
SSH=$(ucloud jobs ssh $JID --print-only)            # "ssh ucloud@ssh.cloud.sdu.dk -p PORT"
$SSH 'hostname; nvidia-smi -L'                      # smoke
$SSH 'setsid nohup /work/hpc/vLLM/.../prov.sh > prov.log 2>&1 < /dev/null &'   # setsid pattern UNCHANGED
ucloud jobs stop $JID                               # no UI, no hold-to-confirm
```

Template-job bootstrap: create ONE known-good job in UI (correct app version, folder attach via dropdown-row click, SSH enabled), note its id, forever clone via `--from`. Store id in repo: `configs/template_job_id`.

## What does NOT change (still runbook-governed)

- In-box persistence: `setsid nohup ... &` + log tailing; UCloud kills session-owned tmux on disconnect. CLI doesn't fix this — nothing can, it's server-side session teardown.
- pkill foot-guns (bracket trick, never compound patterns) — write ops to script, scp, bash.
- venv shim, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, CUDA_HOME toolchain exports.
- Watchers: ssh-grep loop in `tmux_job start(notify_on_completion)` (me) or local while-loop (you). **TTL-guard every watcher** (`END=$((SECONDS+3600)); [ $SECONDS -lt $END ] || exit 1`) — stale watchers on dead ports otherwise poll forever (6 zombies found 2026-08-30). Verified 2026-08-30: marker exit + notify works; remote pipeline (setsid+marker) done in 3.5 min/arm.

## Notes

- rlrs uses undocumented web endpoints ("no stable public API contract") → pin version in coworker docs; if a cmd breaks, `ucloud request GET/POST ...` escape hatch or UI fallback.
- A Go CLI session log exists at `~/.ucloud/cli.log` (tried 2026-08-30, no session configured) — likely the official SDU CLI; not evaluated, ignore unless rlrs breaks.
- TODO (needs live login): validate `submit --from` payload correctness for our mounts, `jobs ssh` port resolution on MIG jobs.
