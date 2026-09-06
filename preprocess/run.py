#!/usr/bin/env python3
"""Preprocess runner: standardise + merge + dedupe the four input kinds.

Output (per donor):
    out/<donor>/<platform>/<OTHER>.json        texts/urls/reactions (media refs only)
    out/<donor>/<platform>/<OTHER>__media/     media bytes, only with --with-media
    out/<donor>/report.json                    per-OTHER stats + source report

Usage:
    uv run python preprocess/run.py [--donor alice-01] [--with-media]
Reads paths from env vars or defaults below.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
import time
from pathlib import Path

# Never write __pycache__ anywhere (uv runs keep CPython's import cache off).
sys.dont_write_bytecode = True

sys.path.insert(0, str(Path(__file__).parent))

from format import Reaction, Thread, enrich_msg, iso_ms  # noqa: E402
from merge import dedupe_threads, finalise, filter_threads, merge_threads, pick_owner  # noqa: E402
from parsers.classic import read_classic  # noqa: E402
from parsers.e2ee import read_e2ee  # noqa: E402
from parsers.wa import read_whatsapp_zip  # noqa: E402

FB_DYI_ZIPDIR = "/Users/e/Library/CloudStorage/GoogleDrive-eliastorjani@gmail.com/My Drive/meta-2026-Sep-03-14-56-41"
FB_E2EE = "/Users/e/Downloads/messages"
IG_DIR = "/Users/e/Library/CloudStorage/GoogleDrive-eliastorjani@gmail.com/My Drive/meta-2026-Sep-03-06-00-19/instagram-eliastorjani-2026-09-03-2BoydNrR/your_instagram_activity/messages"
WA_ZIP = "/Users/e/Downloads/WhatsApp Chat - Rebecca.zip"
# WA: explicit OTHER/SUBJECT mapping when the txt sender names are ambiguous
# (here 'ET' is the participant/subject, 'Rebecca' the chat partner).
WA_OWNER = "ET"
WA_PARTNER = "Rebecca"
WA_ZIP_EXT = ".zip"
OUT = Path(__file__).parent / "out"

SAFE = re.compile(r"[^\w\- ]+")


def _fb_source() -> str:
    # Point at the complete 35-zip set in Google Drive (now fully downloaded).
    # The folder-of-zips branch of read_classic handles every part.
    return FB_DYI_ZIPDIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", default="alice-01")
    ap.add_argument("--with-media", action="store_true", help="copy media bytes into media/")
    ap.add_argument("--media", default="images", choices=["images", "all"],
                    help="which media kinds to copy (default images only)")
    ap.add_argument("--skip-media", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--min-msgs", type=int, default=20, help="min text messages per thread (default 20)")
    # TODO: reactivate when heuristics are settled — see merge.filter_threads.
    # ap.add_argument("--min-active-days", type=int, default=2, ...)
    # ap.add_argument("--min-days", type=int, default=7, ...)
    args = ap.parse_args()

    donor = args.donor
    out_root = OUT / donor
    t0 = time.time()
    report: dict = {"donor": donor, "sources": {}, "merged": {}, "others": {}}
    merged_all: dict[str, list[Thread]] = {}

    print("fb: scanning DYI…", flush=True)
    fb_dyi_src = _fb_source()
    dyi, st_dyi = read_classic(fb_dyi_src, "fb")
    e2ee_threads, st_e2ee = read_e2ee(FB_E2EE)
    st_dyi.archives = len(list(Path(fb_dyi_src).glob("*.zip")))
    report["sources"]["fb-dyi"] = _stats(st_dyi)
    report["sources"]["fb-e2ee"] = _stats(st_e2ee)

    fb_threads = merge_threads(dyi + e2ee_threads)
    fb_threads, fb_dropped = dedupe_threads(fb_threads)
    owner = pick_owner(fb_threads + e2ee_threads)
    fb_threads, fb_filtered = filter_threads(fb_threads, args.min_msgs, owner)
    fb_threads = finalise(fb_threads, "fb", owner)
    report["merged"]["fb"] = {"threads": len(fb_threads),
                              "messages": sum(len(t.messages) for t in fb_threads),
                              "dedupe_dropped": fb_dropped, "owner_hint": owner,
                              "filter_dropped": fb_filtered}
    merged_all["fb"] = fb_threads

    print("ig: scanning…", flush=True)
    ig_threads, st_ig = read_classic(IG_DIR, "ig")
    report["sources"]["ig"] = _stats(st_ig)
    ig_threads = merge_threads(ig_threads)
    ig_threads, ig_dropped = dedupe_threads(ig_threads)
    ig_owner = pick_owner(ig_threads)
    ig_threads, ig_filtered = filter_threads(ig_threads, args.min_msgs, ig_owner)
    ig_threads = finalise(ig_threads, "ig", ig_owner)
    report["merged"]["ig"] = {"threads": len(ig_threads),
                              "messages": sum(len(t.messages) for t in ig_threads),
                              "dedupe_dropped": ig_dropped, "filter_dropped": ig_filtered}
    merged_all["ig"] = ig_threads

    print("wa: scanning…", flush=True)
    # WA uses explicit subject/partner (txt sender names can alias).
    wa_thread, st_wa = read_whatsapp_zip(WA_ZIP)
    if wa_thread:
        wa_thread.participants = [WA_OWNER, WA_PARTNER]
        wa_thread.partner = WA_PARTNER
    report["sources"]["wa"] = _stats(st_wa)
    wa_list = [wa_thread] if wa_thread else []
    if wa_list:
        wa_list, wa_dropped = dedupe_threads(wa_list)
        wa_list, wa_filtered = filter_threads(wa_list, args.min_msgs, "ET")
        wa_list = finalise(wa_list, "wa", "ET")
        report["merged"]["wa"] = {"threads": len(wa_list),
                                  "messages": sum(len(t.messages) for t in wa_list),
                                  "dedupe_dropped": wa_dropped, "filter_dropped": wa_filtered}
        merged_all["wa"] = wa_list
    # ---- per-OTHER split + write --------------------------------------------
    # media zip archives cache (index -> path), read lazily at copy time
    fb_zips: list[Path] = list(Path(fb_dyi_src).glob("*.zip")) if Path(fb_dyi_src).is_dir() else []

    for plat, threads in merged_all.items():
        zips = [Path(WA_ZIP)] if plat == "wa" else (fb_zips if plat == "fb" else [])
        _write_platform(out_root, plat, threads, args, zips)

    report["others"] = {plat: len(ts) for plat, ts in merged_all.items()}
    report["timing_s"] = round(time.time() - t0, 1)
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nartefacts in {out_root}")


def _write_platform(out_root: Path, plat: str, threads: list[Thread], args, zips: list[Path]) -> None:
    plat_dir = out_root / plat
    plat_dir.mkdir(parents=True, exist_ok=True)
    media_dir = out_root / "media"  # one shared store per donor, all platforms
    if args.with_media:
        media_dir.mkdir(exist_ok=True)
    for t in threads:
        if not t.partner:
            continue  # no OTHER (owner-only thread) — skip
        other = SAFE.sub("_", t.partner).strip("_") or "unknown"
        target = plat_dir / f"{other}.json"
        if target.exists():
            continue
        enrich_all(t)
        if args.with_media:
            _copy_media(t, media_dir, zips, args.media)
        payload = {
            "id": t.id, "platform": t.platform, "partner": t.partner,
            "participants": t.participants, "sources": t.sources,
            "first": iso_ms(t.first_ms), "last": iso_ms(t.last_ms),
            "media_dir": "media",
            "messages": [
                {
                    "ts_iso": iso_ms(m.ts), "ts_ms": m.ts, "sender": m.sender,
                    "text": m.text, "urls": m.urls,
                    "reactions": [{"emoji": r.emoji, "actor": r.actor} for r in m.reactions],
                    "media": [{"name": md.uri, "orig": md.orig, "kind": md.kind, "missing": md.missing} for md in m.media],
                    "system": m.system,
                }
                for m in t.messages
            ],
        }
        target.write_text(json.dumps(payload, indent=1, ensure_ascii=False))


def enrich_all(t: Thread) -> None:
    for m in t.messages:
        enrich_msg(m)


def _copy_media(t: Thread, media_dir: Path, zips: list[Path], kinds: str = "all") -> None:
    """Resolve each media locator and copy bytes into the shared media store.

    Names are content-addressed (sha1 of bytes) so the same image sent to two
    people lives once; the placeholder in each chat points at the same file.
    kinds: 'all' copies every kind; 'images' copies only photos.
    """
    import hashlib

    for m in t.messages:
        for md in m.media:
            if md.missing or not md.loc:
                continue
            if kinds == "images" and md.kind != "photos":
                continue
            try:
                if "file" in md.loc and md.loc["file"]:
                    data = Path(md.loc["file"]).read_bytes()
                elif "zip" in md.loc:
                    zf = zips[md.loc["zip"]]
                    with zipfile.ZipFile(zf) as z:
                        data = z.read(md.loc["entry"])
                else:
                    continue
            except Exception:
                md.missing = True
                continue
            ext = Path(md.uri).suffix or ".bin"
            name = hashlib.sha1(data).hexdigest()[:16] + ext
            out = media_dir / name
            if not out.exists():
                out.write_bytes(data)
            md.uri = name


def _stats(s) -> dict:
    return {
        "archives": s.archives, "raw_threads": s.raw_threads,
        "group_threads_dropped": s.group_dropped, "messages": s.messages,
        "text_chars": s.text_chars, "media": s.media, "escapes_repaired": s.escapes_repaired,
    }


if __name__ == "__main__":
    main()