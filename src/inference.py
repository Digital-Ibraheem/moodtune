"""Model loading + predict_array. Shared between Gradio and FastAPI."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHECKPOINT_DIR, LABELS, MAX_SAMPLES, SAMPLE_RATE  # noqa: E402
from src.dataset import normalize_waveform  # noqa: E402
from src.model import build_model  # noqa: E402

_model = None
_device = None
_trained = False


def get_model():
    global _model, _device, _trained
    if _model is None:
        _device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model, _ = build_model(num_labels=len(LABELS))
        ckpt_path = CHECKPOINT_DIR / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["state_dict"])
            _trained = True
            print(f"[inference] loaded {ckpt_path}  (val_acc={ckpt.get('val_accuracy', 'n/a')})")
        else:
            print("[inference] no checkpoint found — running with random head")
        model.to(_device).eval()
        _model = model
    return _model, _device


def is_trained() -> bool:
    get_model()
    return _trained


def _prep_waveform(sr: int, wav: np.ndarray) -> torch.Tensor:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    t = torch.from_numpy(wav).unsqueeze(0)
    if sr != SAMPLE_RATE:
        t = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(t)
    t = t.squeeze(0)
    # Identical silence-trim + RMS-normalize as training so the distribution
    # the model sees at serve time matches what it learned from.
    t = torch.from_numpy(normalize_waveform(t.numpy()))
    if t.shape[0] < MAX_SAMPLES:
        t = torch.nn.functional.pad(t, (0, MAX_SAMPLES - t.shape[0]))
    else:
        t = t[:MAX_SAMPLES]
    return t


def predict_array(sr: int, wav: np.ndarray) -> dict:
    model, device = get_model()
    t = _prep_waveform(sr, wav).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_values=t).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}
