"""Gradio demo for MoodTune. Classifier head is untrained — predictions are not meaningful."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import LABELS, MAX_SAMPLES, SAMPLE_RATE  # noqa: E402
from src.model import build_model  # noqa: E402

DISCLAIMER = (
    "### MoodTune — Speech Emotion Recognition\n"
    "> ⚠️ **Work in progress.** The classifier head is currently **untrained** — "
    "predictions are not meaningful yet. Training is pending on a GPU machine. "
    "See the README for methodology."
)

ABOUT = (
    "MoodTune fine-tunes Wav2Vec2 on RAVDESS for 4-class emotion recognition "
    "(neutral / happy / sad / angry) and evaluates cross-corpus generalization on CREMA-D. "
    "Repo: https://github.com/Digital-Ibraheem/moodtune"
)

_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _ = build_model(num_labels=len(LABELS))
        model.to(_device).eval()
        _model = model
    return _model, _device


def _to_waveform(audio) -> torch.Tensor:
    if audio is None:
        raise gr.Error("please provide audio via upload or microphone")
    sr, wav = audio
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    peak = float(np.max(np.abs(wav))) or 1.0
    if peak > 1.0:
        wav = wav / peak
    t = torch.from_numpy(wav).unsqueeze(0)
    if sr != SAMPLE_RATE:
        t = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(t)
    t = t.squeeze(0)
    if t.shape[0] < MAX_SAMPLES:
        t = torch.nn.functional.pad(t, (0, MAX_SAMPLES - t.shape[0]))
    else:
        t = t[:MAX_SAMPLES]
    return t


def predict(audio) -> dict:
    model, device = _get_model()
    wav = _to_waveform(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_values=wav).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
    return {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}


def build_app() -> gr.Blocks:
    with gr.Blocks(title="MoodTune") as app:
        gr.Markdown(DISCLAIMER)
        with gr.Row():
            audio_in = gr.Audio(sources=["upload", "microphone"], type="numpy", label="Audio")
        btn = gr.Button("Analyze", variant="primary")
        label_out = gr.Label(num_top_classes=4, label="Predicted emotion probabilities")
        btn.click(fn=predict, inputs=audio_in, outputs=label_out)
        with gr.Accordion("About", open=False):
            gr.Markdown(ABOUT)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app().launch(share=args.share)


if __name__ == "__main__":
    main()
