"""Audeering wav2vec2-large-robust MSP-Podcast backend.

Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim (~1.2GB).
Trained on MSP-Podcast — natural, non-acted podcast speech with real emotional
variability and real-world acoustic conditions. This is the best off-the-shelf
SER backbone available for laptop-mic audio.

The model outputs 3 continuous dimensions in [0, 1]:
  - arousal (calm ↔ active)
  - dominance (submissive ↔ dominant)
  - valence (negative ↔ positive)

We map that to {neutral, happy, sad, angry} by placing category prototypes in
the (arousal, valence) plane and taking a softmax over negative distance. This
lets "neutral" have real probability mass when the speaker is genuinely flat,
which is exactly what the SUPERB categorical model was failing to do.

License note: audeering's weights are CC-BY-NC-SA-4.0 — fine for a portfolio
project, requires a non-commercial badge if re-distributed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model, Wav2Vec2PreTrainedModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SAMPLE_RATE  # noqa: E402

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


class _RegressionHead(nn.Module):
    """Replica of audeering's head — a tanh-activated MLP to 3 outputs."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Replicates the EmotionModel class from the audeering HF model card."""

    # Newer transformers call self.all_tied_weights_keys during init_weights;
    # our subclass has no tied weights. Expose an empty dict to satisfy the API.
    all_tied_weights_keys = {}
    _tied_weights_keys: list[str] = []

    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0].mean(dim=1)  # time-pool
        logits = self.classifier(hidden)
        return hidden, logits


# Prototype positions in (arousal, valence) space, hand-tuned against real
# laptop-mic behaviour: the MSP-Podcast model tends to report moderate arousal
# for at-desk speech (no projection, no studio gain), so angry lives at
# arousal ~0.6 — not the 0.78 a studio shout would produce. Neutral is nudged
# off-center so it doesn't swallow every mid-range reading.
_PROTOS: dict[str, tuple[float, float]] = {
    "neutral": (0.48, 0.60),
    "happy":   (0.70, 0.80),
    "sad":     (0.30, 0.26),
    "angry":   (0.62, 0.22),
}
_LABELS = ["neutral", "happy", "sad", "angry"]
# Lower → sharper decisions. 0.10 makes the winning class more decisive
# without collapsing the distribution.
_TEMPERATURE = 0.10

_model: EmotionModel | None = None
_fe: Wav2Vec2FeatureExtractor | None = None
_device: torch.device | None = None


def get_model():
    global _model, _fe, _device
    if _model is None:
        _device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[audeering] loading {MODEL_NAME} on {_device} (~1.2GB, first run will download)")
        # Regression model — no tokenizer, only a feature extractor.
        _fe = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

        # Loading here is hand-rolled for two reasons:
        #   1) audeering's config.json has vocab_size=null — newer transformers
        #      versions reject that at dataclass validation.
        #   2) from_pretrained() on our custom EmotionModel trips on
        #      `all_tied_weights_keys` (internal API added in recent releases).
        # Easiest path: download the config + weights, patch the config, init
        # EmotionModel from the config, and load the state_dict directly.
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as load_safetensors

        cfg_path = hf_hub_download(MODEL_NAME, "config.json")
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        if cfg_dict.get("vocab_size") is None:
            cfg_dict["vocab_size"] = 32
        config = Wav2Vec2Config(**cfg_dict)

        try:
            weights_path = hf_hub_download(MODEL_NAME, "model.safetensors")
            state_dict = load_safetensors(weights_path)
        except Exception:
            weights_path = hf_hub_download(MODEL_NAME, "pytorch_model.bin")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        m = EmotionModel(config)
        missing, unexpected = m.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"[audeering] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print("  first missing:", missing[:3])
        m = m.to(_device).eval()
        _model = m
        print("[audeering] loaded")
    return _model, _fe, _device


def is_trained() -> bool:
    get_model()
    return True


def _map_to_categories(arousal: float, valence: float) -> dict:
    """Softmax over negative squared distance to each class prototype."""
    scores = np.array(
        [
            -((arousal - a) ** 2 + (valence - v) ** 2) / _TEMPERATURE
            for (a, v) in (_PROTOS[lbl] for lbl in _LABELS)
        ]
    )
    scores -= scores.max()
    exp = np.exp(scores)
    probs = exp / exp.sum()
    return {lbl: float(p) for lbl, p in zip(_LABELS, probs)}


def predict_array(sr: int, wav: np.ndarray) -> dict:
    model, fe, device = get_model()

    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    if sr != SAMPLE_RATE:
        t = torch.from_numpy(wav).unsqueeze(0)
        t = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(t)
        wav = t.squeeze(0).numpy()

    inputs = fe(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        _, dims = model(input_values=input_values)
    dims = dims.squeeze(0).cpu().numpy()  # (arousal, dominance, valence)
    arousal, _dom, valence = float(dims[0]), float(dims[1]), float(dims[2])
    print(f"[audeering] arousal={arousal:.3f} dominance={_dom:.3f} valence={valence:.3f}")

    return _map_to_categories(arousal, valence)
