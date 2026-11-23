"""Pull a donation from the UCloud donation drive to wherever the analysis runs.

The donation project drive is a separate UCloud project from the analysis workspace, on purpose: the
relay's internet-facing token can only see donations, and a compromise there cannot touch research data.
Data crosses by *pull*, one direction, with its own credentials — this script.

UCloud's download is a two-step ticket like the upload the browser does (documented in
thesis-deid/docs/ucloud-upload-protocol.md):

    POST /api/files/download  {"items":[{"id":"/<drive>/<participant>/bundle-….zip"}]}
    → {"responses":[{"endpoint":"https://k8s.cloud.sdu.dk/ucloud/ucloud/download?token=…"}]}
    GET  <endpoint>   → the bytes

The token in that URL is single-use and short-lived, and is never printed by this script.

    UCLOUD_TOKEN=… UCLOUD_PROJECT=… uv run python -B pull_donation.py --list
    UCLOUD_TOKEN=… UCLOUD_PROJECT=… uv run python -B pull_donation.py --participant p999-… --out data/p999

Env, not flags, for the secrets: a token in `argv` lands in the shell history and in `ps`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

API = os.environ.get("UCLOUD_API", "https://cloud.sdu.dk/api")
DRIVE_ROOT = os.environ.get("UCLOUD_DRIVE_ROOT", "")


def _headers() -> dict[str, str]:
    token = os.environ.get("UCLOUD_TOKEN")
    project = os.environ.get("UCLOUD_PROJECT")
    if not token or not project:
        raise SystemExit("set UCLOUD_TOKEN and UCLOUD_PROJECT (see the study's shared/.env)")
    return {"Authorization": f"Bearer {token}", "Project": project}


def _get(path: str, params: dict | None = None) -> dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(body).encode(),
        headers={**_headers(), "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def list_participants() -> list[dict]:
    """Participants are folders; a folder whose name matches the id shape is one of ours."""
    items = _get("/files/browse", {"path": DRIVE_ROOT, "itemsPerPage": 250}).get("items", [])
    out = []
    for item in items:
        name = str(item.get("id", "")).split("/")[-1]
        if name.startswith("p") and "-" in name and (item.get("status") or {}).get("type") == "DIRECTORY":
            files = _get("/files/browse", {"path": f"{DRIVE_ROOT}/{name}", "itemsPerPage": 250}).get("items", [])
            bundles = [f for f in files if str(f.get("id", "")).endswith(".zip")]
            out.append({"participant": name, "bundles": sorted(str(b["id"]) for b in bundles),
                        "surveys": [str(f["id"]) for f in files if "/survey" in str(f.get("id", ""))]})
    return out


def download(remote: str, target: Path) -> tuple[int, str]:
    """Fetch one file through a download ticket. Returns (bytes, sha256)."""
    ticket = _post("/files/download", {"items": [{"id": remote}]})["responses"][0]["endpoint"]
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with urllib.request.urlopen(ticket, timeout=1800) as response, target.open("wb") as sink:
        while chunk := response.read(1 << 20):
            sink.write(chunk)
            digest.update(chunk)
            written += len(chunk)
    return written, digest.hexdigest()


def unpack(archive: Path, into: Path) -> dict:
    """Unpack and say what arrived, so a mismatch with the receipt is visible immediately."""
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise SystemExit(f"corrupt member in {archive.name}: {bad}")
        bundle.extractall(into)
    report = json.loads((into / "report.json").read_text(encoding="utf-8"))
    return {
        "render_version": report.get("render_version"),
        "donor_id": report.get("donor_id"),
        "conversations": (report.get("volume") or {}).get("conversations"),
        "prompt_tokens_M": ((report.get("volume") or {}).get("prompt") or {}).get("tokens_M"),
        "kept_for_review": (report.get("kept_for_review") or {}).get("enabled"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="what is on the drive, and from whom")
    parser.add_argument("--participant", help="the folder to pull")
    parser.add_argument("--out", type=Path, help="where to unpack (one directory per participant)")
    parser.add_argument("--keep-zip", action="store_true", help="leave the archive beside the unpacked tree")
    args = parser.parse_args(argv)

    if args.list or not args.participant:
        for row in list_participants():
            print(f"{row['participant']:26s} bundles={len(row['bundles'])} surveys={len(row['surveys'])}")
        return 0

    if not args.out:
        parser.error("--out is required when pulling a participant")
    row = next((r for r in list_participants() if r["participant"] == args.participant), None)
    if row is None:
        raise SystemExit(f"no folder for {args.participant} on {DRIVE_ROOT}")
    if not row["bundles"]:
        raise SystemExit(f"{args.participant} has uploaded no bundle")

    remote = row["bundles"][-1]  # names are timestamped, so the last one sorts newest
    archive = args.out.with_suffix(".zip")
    size, sha = download(remote, archive)
    print(f"downloaded {remote.split('/')[-1]}: {size:,} bytes · sha256 {sha}")

    summary = unpack(archive, args.out)
    print(json.dumps(summary, indent=1))
    if not args.keep_zip:
        archive.unlink()
    else:
        print(f"kept {archive}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
