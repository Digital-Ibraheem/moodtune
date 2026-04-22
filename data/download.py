"""Download RAVDESS and CREMA-D audio corpora into data/raw/.

Licensing notes:
    RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song
        (Livingstone & Russo, 2018). Released under CC BY-NC-SA 4.0.
        Zenodo DOI: 10.5281/zenodo.1188976.
    CREMA-D — Crowd-sourced Emotional Multimodal Actors Dataset
        (Cao et al., 2014). Released under the Open Database License.
        Source: https://github.com/CheyneyComputerScience/CREMA-D.

Both corpora are free for non-commercial research. This script only
downloads files; redistribution rules still apply to derivative work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR  # noqa: E402

RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
)
# CREMA-D stores WAVs via Git LFS. The raw.githubusercontent endpoint transparently
# resolves the LFS pointers, so we pull files individually in parallel. The Git Tree
# API returns every path in one response, which avoids paginated rate-limit issues.
CREMA_TREE_API = (
    "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/git/trees/master?recursive=1"
)
# Must use github.com/.../raw/ — that endpoint resolves Git LFS pointers server-side.
# raw.githubusercontent.com serves the pointer file verbatim instead.
CREMA_RAW_BASE = (
    "https://github.com/CheyneyComputerScience/CREMA-D/raw/master/AudioWAV"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def download_ravdess(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    wavs = list(target_dir.rglob("*.wav"))
    if len(wavs) >= 1000:
        print(f"[ravdess] already present ({len(wavs)} wavs), skipping")
        return

    zip_path = target_dir / "Audio_Speech_Actors_01-24.zip"
    if not zip_path.exists():
        _download(RAVDESS_URL, zip_path)

    print(f"[ravdess] extracting {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)
    zip_path.unlink()
    print(f"[ravdess] done: {len(list(target_dir.rglob('*.wav')))} wavs")


def _list_crema_files(limit: int | None = None) -> list[str]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(CREMA_TREE_API, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read())
    if payload.get("truncated"):
        print("[crema] WARN: git tree response was truncated")
    names = [
        Path(t["path"]).name
        for t in payload.get("tree", [])
        if t.get("path", "").startswith("AudioWAV/") and t["path"].endswith(".wav")
    ]
    return names[:limit] if limit is not None else names


def _fetch_one(name: str, audio_dir: Path) -> tuple[str, bool, str]:
    dest = audio_dir / name
    if dest.exists() and dest.stat().st_size > 1024:
        return name, True, "cached"
    url = f"{CREMA_RAW_BASE}/{name}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp, open(dest, "wb") as out:
            shutil.copyfileobj(resp, out)
        return name, True, "ok"
    except Exception as exc:  # noqa: BLE001
        return name, False, str(exc)


def download_crema_d(target_dir: Path, sample_only: bool = False, workers: int = 24) -> None:
    """Parallel per-file pull from raw.githubusercontent (LFS-resolved)."""
    audio_dir = target_dir / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for p in audio_dir.glob("*.wav") if p.stat().st_size > 1024]
    min_expected = 500 if sample_only else 7000
    if len(existing) >= min_expected:
        print(f"[crema] already present ({len(existing)} wavs), skipping")
        return

    limit = 500 if sample_only else None
    print(f"[crema] listing files (sample_only={sample_only})")
    names = _list_crema_files(limit=limit)
    print(f"[crema] fetching {len(names)} wavs with {workers} workers")

    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, n, audio_dir) for n in names]
        for fut in as_completed(futures):
            _, ok, _ = fut.result()
            done += 1
            if not ok:
                failed += 1
            if done % 200 == 0:
                print(f"[crema] {done}/{len(names)} ({failed} failed)")
    ok_wavs = [p for p in audio_dir.glob("*.wav") if p.stat().st_size > 1024]
    print(f"[crema] done: {len(ok_wavs)} wavs ({failed} failed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RAVDESS and CREMA-D corpora")
    parser.add_argument("--ravdess-only", action="store_true")
    parser.add_argument("--crema-only", action="store_true")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Grab only ~500 CREMA-D files (faster demo mode)",
    )
    args = parser.parse_args()

    if not args.crema_only:
        download_ravdess(RAW_DIR / "ravdess")
    if not args.ravdess_only:
        download_crema_d(RAW_DIR / "crema_d", sample_only=args.sample)


if __name__ == "__main__":
    main()
