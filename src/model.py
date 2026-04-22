"""Wav2Vec2 + classification head for 4-way emotion recognition."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LABELS, MODEL_NAME, SAMPLE_RATE  # noqa: E402


def build_model(num_labels: int = 4, freeze_feature_extractor: bool = True):
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    if freeze_feature_extractor:
        model.freeze_feature_encoder()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    return model, feature_extractor


def count_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"parameters: total={total:,} trainable={trainable:,} frozen={frozen:,}")
    return {"total": total, "trainable": trainable, "frozen": frozen}


if __name__ == "__main__":
    print(f"building {MODEL_NAME} with {len(LABELS)} labels: {LABELS}")
    model, fe = build_model(num_labels=len(LABELS))
    count_parameters(model)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, SAMPLE_RATE * 4)
        out = model(input_values=dummy)
    print(f"forward OK — logits shape: {tuple(out.logits.shape)}")
