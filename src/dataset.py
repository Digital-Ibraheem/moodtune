"""PyTorch Dataset for emotion audio with on-the-fly resampling and padding."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MAX_SAMPLES, PROCESSED_DIR, SAMPLE_RATE  # noqa: E402


class EmotionAudioDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, split: Optional[str] = None):
        if split is not None:
            manifest = manifest[manifest["split"] == split]
        self.df = manifest.reset_index(drop=True)
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        if orig_sr == SAMPLE_RATE:
            return wav
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
        return self._resamplers[orig_sr](wav)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        data, sr = sf.read(row["filepath"], dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        wav = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)
        wav = self._resample(wav, sr).squeeze(0)

        if wav.shape[0] < MAX_SAMPLES:
            wav = torch.nn.functional.pad(wav, (0, MAX_SAMPLES - wav.shape[0]))
        else:
            wav = wav[:MAX_SAMPLES]

        return {
            "input_values": wav,
            "label": int(row["label_id"]),
            "speaker_id": row["speaker_id"],
            "corpus": row["corpus"],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "input_values": torch.stack([b["input_values"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "speaker_ids": [b["speaker_id"] for b in batch],
        "corpora": [b["corpus"] for b in batch],
    }


def make_dataloader(dataset: EmotionAudioDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate,
        num_workers=0,
    )


if __name__ == "__main__":
    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"manifest not found at {manifest_path}. "
            "Run `python -m data.download && python -m data.prepare` first."
        )
    manifest = pd.read_csv(manifest_path)
    ds = EmotionAudioDataset(manifest, split="train")
    print(f"train dataset size: {len(ds)}")
    loader = make_dataloader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"input_values: {batch['input_values'].shape} dtype={batch['input_values'].dtype}")
    print(f"labels: {batch['labels'].tolist()}")
    print(f"speakers: {batch['speaker_ids']}")
    print(f"corpora: {batch['corpora']}")
