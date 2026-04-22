"""Drop-in inference backend using the SUPERB ER model.

Why this exists: our RAVDESS+CREMA-D-trained model overfits acted studio speech
and struggles on a laptop mic. `superb/wav2vec2-base-superb-er` was fine-tuned
on IEMOCAP (dyadic dialogue) and covers exactly our 4 classes. It ships on HF;
we load it once and expose the same `predict_array` signature the FastAPI
server already uses, so the swap is one import line.

The local-fine-tune code path (src/inference.py) is kept — it's the portfolio
story, and it powers the benchmark numbers in the README. This module is only
used at serve time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_RATE  # noqa: E402

MODEL_NAME = "superb/wav2vec2-base-superb-er"

# SUPERB ER was trained on IEMOCAP with these label strings. Map to our
# frontend-facing keys so the UI doesn't have to care which backend served it.
SUPERB_LABEL_MAP = {
    "neu": "neutral",
    "hap": "happy",
    "sad": "sad",
    "ang": "angry",
}

_model = None
_feature_extractor = None
_device = None


def get_model():
    global _model, _feature_extractor, _device
    if _model is None:
        _device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[superb] loading {MODEL_NAME} on {_device}")
        _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME).to(_device).eval()
        print(f"[superb] labels: {_model.config.id2label}")
    return _model, _feature_extractor, _device


def is_trained() -> bool:
    get_model()
    return True


def predict_array(sr: int, wav: np.ndarray) -> dict:
    model, fe, device = get_model()

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # Resample to 16 kHz if needed — model was trained at that rate.
    if sr != SAMPLE_RATE:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(t)
        wav = t.squeeze(0).numpy()

    inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    id2label = model.config.id2label
    out = {lbl: 0.0 for lbl in ("neutral", "happy", "sad", "angry")}
    for idx, p in enumerate(probs):
        raw = id2label[idx].lower()
        key = SUPERB_LABEL_MAP.get(raw, raw)
        if key in out:
            out[key] = float(p)
    return out
