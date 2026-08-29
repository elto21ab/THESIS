# UCloud SSH access — setup & runbook

Verified on: B200 MIG machines, Terminal-Ubuntu app (Aug 2026). Applies to any interactive job.

## 1. One-time key setup (done)

- Local key: `~/.ssh/id_ed25519`, registered in UCloud → Resources → SSH keys as **"Elias MacBook Pro - pi testing"** (fingerprint `SHA256:Sx8uENOsBxtIRj01WZPXNrMxx3SWDQn+w5ZIGYORBxU`).
- **Gotcha #1 (passphrase/agent):** the private key is passphrase-protected. Every new Mac session (reboot, new shell) needs:
  ```bash
  ssh-add --apple-use-keychain ~/.ssh/id_ed25519
  ```
  Symptom when forgotten: `Server accepts key` then `Permission denied (publickey)` — misleading; the *client* can't decrypt the key, the server is fine.
- Enable ssh-agent via Keychain in `~/.ssh/config` if desired:
  ```
  Host ssh.cloud.sdu.dk
    UseKeychain yes
    AddKeysToAgent yes
    IdentityFile ~/.ssh/id_ed25519
  ```
  (Note: config currently has stale `Port` lines — always pass `-p` explicitly, it wins.)

## 2. Per-job access (every run)

1. Submit job with **Enable SSH server** checked (Terminal-Ubuntu app → "Configure SSH access").
2. Port is **dynamic per job**. Read it from the job page → SSH panel (e.g. `ssh ucloud@ssh.cloud.sdu.dk -p 2124`). Old ports from previous jobs don't route to the new job.
3. Connect:
  ```bash
  ssh ucloud@ssh.cloud.sdu.dk -p <PORT> 'hostname; nvidia-smi -L'
  ```
4. Key added after job launch? UCloud does NOT inject keys into running jobs → stop + relaunch job.

### Debug ladder (when "Permission denied")
1. `ssh-add -l` empty → run the `ssh-add --apple-use-keychain` command above (99% of cases).
2. Confirm port from the *current* job page, not memory/history.
3. Fallback: drive the **web terminal** (ttyd, canvas-rendered — invisible to DOM snapshots; drive blind via keyboard events).
4. Server-side check via web terminal: `journalctl -u ssh` / `~/.ssh/authorized_keys` perms (700/600).

## 3. Remote tmux — long jobs

UCloud kills a tmux server started from an SSH session once that session closes (no linger). Two options:

**A. setsid (preferred for scripted pipelines):**
```bash
ssh ucloud@... -p PORT 'setsid nohup /work/LCPP_OffloadTesting/qwen38/prov.sh > prov.log 2>&1 < /dev/null &'
```
Survives disconnect; monitor by tailing the log from later ssh calls.

**B. Interactive tmux** (Enable tmux app param = true, or manual `tmux new -d -s w`) only if you keep one SSH session alive as anchor.

**Gotcha #2 (pkill foot-gun):** `pkill -f "vllm serve"` in a compound command whose own cmdline contains the literal pattern (incl. a preceding `pgrep -af "bench|vllm"`) matches *its own ssh/bash process* and kills the session (exit 255). Use bracket trick `pkill -f "[v]llm serve"` AND never mix a `pgrep` pattern containing the same literal into the same command line.

## 4. Conventions

- Persistent storage: `/work/LCPP_OffloadTesting` (mounted automatically when selected in job creation — verify via `JobParameters.json` / `ls /work`).
- Per-experiment dirs: `/work/LCPP_OffloadTesting/<exp>/` with `prov.sh`, `bench.sh`, `results/`, `hf/` (HF cache).
- `~/.ssh/config` on the Mac contains historical per-job ports; ignore them, always pass `-p <current-port>`.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` required on MIG (torch NVML assert); values must be capitalized (`False` not `false` — this torch build's parser is case-strict).
- vLLM JIT needs nvcc + dev headers; toolchain persisted at `/work/LCPP_OffloadTesting/runtime/cuda-13/` — export `CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` before any serve (see QWEN38_NVFP4_EXPERIMENT.md pre-flight checklist).
- `pkill -f` second instance: never in a compound command whose own cmdline contains the pattern from an earlier `pgrep -af` in the same line — the pgrep pattern text itself matches and kills the session.

## 2026-08-29 additions
- Remote restructured: everything now under `/work/LCPP_OffloadTesting/hpc/` (models, toolchain, venvs, data, experiments, results). `llama-inference/` folded into `hpc/experiments/2026-08-19_llama_first/` and its source emptied.
- Filesystem-only maintenance: cheap `terminal-ubuntu` job on `cpu-amd-zen5-1-vcpu` (1 core-hour) w/ folders attached does the job. "Add folder" needs the folder-row click in the picker dialog to actually attach.
- Stop button is **hold-to-confirm**: `pb hold <session> <ref> --ms 2500` (new patchright CLI command added this date; plain click only arms it).
