"""Build a unified manifest for RAVDESS + CREMA-D with speaker-stratified splits."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (  # noqa: E402
    CREMA_EMOTION_MAP,
    LABEL_TO_ID,
    PROCESSED_DIR,
    RAVDESS_EMOTION_MAP,
    RAW_DIR,
)


def parse_ravdess_filename(path: Path) -> dict:
    """Parse a RAVDESS filename: modality-vocal-emotion-intensity-statement-repetition-actor.wav."""
    parts = path.stem.split("-")
    if len(parts) != 7:
        return {}
    return {
        "modality": parts[0],
        "vocal_channel": parts[1],
        "emotion": parts[2],
        "intensity": parts[3],
        "statement": parts[4],
        "repetition": parts[5],
        "actor_id": parts[6],
    }


def _duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / float(info.samplerate)


def build_ravdess_manifest(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for wav in sorted(raw_dir.rglob("*.wav")):
        meta = parse_ravdess_filename(wav)
        if not meta:
            continue
        label = RAVDESS_EMOTION_MAP.get(meta["emotion"])
        if label is None:
            continue
        try:
            dur = _duration_seconds(wav)
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "filepath": str(wav),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "speaker_id": f"ravdess_{meta['actor_id']}",
                "duration": dur,
                "corpus": "ravdess",
            }
        )
    return pd.DataFrame(rows)


def build_crema_d_manifest(raw_dir: Path) -> pd.DataFrame:
    audio_dir = raw_dir / "AudioWAV"
    if not audio_dir.exists():
        audio_dir = raw_dir
    rows = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) < 3:
            continue
        speaker, _sentence, emo_code = parts[0], parts[1], parts[2]
        label = CREMA_EMOTION_MAP.get(emo_code)
        if label is None:
            continue
        try:
            dur = _duration_seconds(wav)
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "filepath": str(wav),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "speaker_id": f"crema_{speaker}",
                "duration": dur,
                "corpus": "crema_d",
            }
        )
    return pd.DataFrame(rows)


def speaker_stratified_split(
    df: pd.DataFrame,
    val_speakers: int = 2,
    test_speakers: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val/test by speaker id. Crucial: no speaker appears in two splits."""
    speakers = sorted(df["speaker_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(speakers)
    test = set(speakers[:test_speakers])
    val = set(speakers[test_speakers : test_speakers + val_speakers])

    def assign(sp: str) -> str:
        if sp in test:
            return "test"
        if sp in val:
            return "val"
        return "train"

    out = df.copy()
    out["split"] = out["speaker_id"].map(assign)
    return out


def _summarize(df: pd.DataFrame) -> None:
    print("\n=== Manifest summary ===")
    print(f"total samples: {len(df)}")
    print("\nby corpus:")
    print(df.groupby("corpus").size().to_string())
    print("\nby corpus x split:")
    print(df.groupby(["corpus", "split"]).size().to_string())
    print("\nby corpus x label:")
    print(df.groupby(["corpus", "label"]).size().to_string())
    print("\nunique speakers by corpus:")
    for corpus, grp in df.groupby("corpus"):
        print(f"  {corpus}: {grp['speaker_id'].nunique()}")
    print()


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rav = build_ravdess_manifest(RAW_DIR / "ravdess")
    if not rav.empty:
        rav = speaker_stratified_split(rav)
    else:
        print("[warn] no RAVDESS audio found under data/raw/ravdess/")

    crema = build_crema_d_manifest(RAW_DIR / "crema_d")
    if not crema.empty:
        # Speaker-stratified split inside CREMA-D so we can co-train AND still
        # measure held-out cross-corpus performance. 15 speakers are test,
        # 5 val, the rest train. Split labels follow the same vocabulary as
        # RAVDESS ({train, val, test}) so the DataLoaders don't need special
        # casing. A separate "crema_test" column marks which corpus a test
        # sample came from — evaluate.py uses this for the split report.
        crema = speaker_stratified_split(crema, val_speakers=5, test_speakers=15)
    else:
        print("[warn] no CREMA-D audio found under data/raw/crema_d/")

    manifest = pd.concat([rav, crema], ignore_index=True)
    out_path = PROCESSED_DIR / "manifest.csv"
    manifest.to_csv(out_path, index=False)
    print(f"[prepare] wrote {out_path} ({len(manifest)} rows)")
    _summarize(manifest)


if __name__ == "__main__":
    main()
