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
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR  # noqa: E402

RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
)
# Per-file raw URL pattern for CREMA-D AudioWAV.
CREMA_LISTING_API = (
    "https://api.github.com/repos/CheyneyComputerScience/CREMA-D/contents/AudioWAV"
)
CREMA_RAW_BASE = (
    "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/AudioWAV"
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
    """Return AudioWAV filenames via the GitHub contents API (paginated)."""
    names: list[str] = []
    page = 1
    while True:
        url = f"{CREMA_LISTING_API}?per_page=100&page={page}"
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req) as resp:
            import json

            batch = json.loads(resp.read())
        if not batch:
            break
        for entry in batch:
            name = entry.get("name", "")
            if name.endswith(".wav"):
                names.append(name)
        if limit is not None and len(names) >= limit:
            break
        page += 1
    if limit is not None:
        names = names[:limit]
    return names


def download_crema_d(target_dir: Path, sample_only: bool = False) -> None:
    audio_dir = target_dir / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing = list(audio_dir.glob("*.wav"))
    min_expected = 200 if sample_only else 7000
    if len(existing) >= min_expected:
        print(f"[crema] already present ({len(existing)} wavs), skipping")
        return

    limit = 500 if sample_only else None
    print(f"[crema] listing files (sample_only={sample_only})")
    names = _list_crema_files(limit=limit)
    print(f"[crema] downloading {len(names)} wavs")
    for i, name in enumerate(names, 1):
        dest = audio_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        url = f"{CREMA_RAW_BASE}/{name}"
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"[crema] skip {name}: {exc}")
        if i % 50 == 0:
            print(f"[crema] {i}/{len(names)}")
    print(f"[crema] done: {len(list(audio_dir.glob('*.wav')))} wavs")


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
