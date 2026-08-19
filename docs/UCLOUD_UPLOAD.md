# Upload data to a running UCloud job

## Owner: once per uploader

1. Ask uploader to run setup below and send their **public** key.
2. UCloud → Resources → SSH keys → Add SSH key.
3. Launch a job with **Enable SSH server**. A running job may require relaunch to receive a newly added key.
4. Job page → SSH → send uploader the current port. Port changes per job.

## Uploader: one-time setup

Download the script from a trusted pinned commit/release; verify its published SHA-256. Do not use `curl | bash`.

```bash
chmod +x ucloud-upload.sh
./ucloud-upload.sh setup YOUR_NAME
```

Choose a strong passphrase. Send owner only the printed `ssh-ed25519 ...` line. Never send `~/.ssh/ucloud_upload` or passphrase.

## Every upload

Put files in `~/Desktop/data`, then use port supplied by owner:

```bash
./ucloud-upload.sh upload PORT "$HOME/Desktop/data" YOUR_NAME
```

Script creates `/work/llama-inference/data/YOUR_NAME/`, uploads via resumable `rsync`, verifies file count/size, then SSH exits automatically. It never deletes remote files.

Requirements: macOS/Linux, OpenSSH, `rsync`. Windows: use WSL; native PowerShell not supported by this Bash script.

## Security limit

Every key logs in as Unix user `ucloud`; uploaders can access everything that user can. Use only for trusted collaborators. Removing a UI key may not revoke an already-running job: stop job, remove key, relaunch.
