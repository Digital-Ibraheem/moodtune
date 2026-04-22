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
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import RAW_DIR  # noqa: E402

RAVDESS_URL = (
    "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
)
# Single tarball of the full CREMA-D repo — faster than per-file pulls.
CREMA_TARBALL_URL = (
    "https://codeload.github.com/CheyneyComputerScience/CREMA-D/tar.gz/refs/heads/master"
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


def download_crema_d(target_dir: Path, sample_only: bool = False) -> None:
    """Pull CREMA-D via a single tarball and extract only AudioWAV/.

    sample_only keeps only the first 500 wavs after extraction — handy for
    iterating locally without shipping 500MB of audio through the pipeline.
    """
    audio_dir = target_dir / "AudioWAV"
    audio_dir.mkdir(parents=True, exist_ok=True)

    existing = list(audio_dir.glob("*.wav"))
    min_expected = 200 if sample_only else 7000
    if len(existing) >= min_expected:
        print(f"[crema] already present ({len(existing)} wavs), skipping")
        return

    tar_path = target_dir / "crema_d.tar.gz"
    if not tar_path.exists():
        _download(CREMA_TARBALL_URL, tar_path)

    print(f"[crema] extracting AudioWAV/ from {tar_path}")
    count = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] != "AudioWAV" or not parts[-1].endswith(".wav"):
                continue
            dest = audio_dir / parts[-1]
            if dest.exists() and dest.stat().st_size > 0:
                count += 1
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with open(dest, "wb") as out:
                shutil.copyfileobj(extracted, out)
            count += 1
            if sample_only and count >= 500:
                break
    tar_path.unlink()
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
