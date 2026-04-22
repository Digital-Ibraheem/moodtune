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
from src.config import CHECKPOINT_DIR, LABELS, MAX_SAMPLES, SAMPLE_RATE  # noqa: E402
from src.model import build_model  # noqa: E402

DISCLAIMER_TRAINED = (
    "### MoodTune — Speech Emotion Recognition\n"
    "> 4-class classifier (neutral / happy / sad / angry) — Wav2Vec2 with a fine-tuned head. "
    "Trained on RAVDESS; accuracy is much lower on out-of-distribution speech. "
    "See the README for methodology and honest cross-corpus numbers."
)
DISCLAIMER_UNTRAINED = (
    "### MoodTune — Speech Emotion Recognition\n"
    "> ⚠️ No checkpoint found at `checkpoints/best.pt` — running with a random classifier head. "
    "Predictions are **not meaningful**. Train first: `python -m src.train`."
)

ABOUT = (
    "MoodTune fine-tunes Wav2Vec2 on RAVDESS for 4-class emotion recognition "
    "(neutral / happy / sad / angry) and evaluates cross-corpus generalization on CREMA-D. "
    "Repo: https://github.com/Digital-Ibraheem/moodtune"
)

_model = None
_device = None
_trained = False


def _get_model():
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
            print(f"[demo] loaded {ckpt_path}  (val_acc={ckpt.get('val_accuracy', 'n/a')})")
        else:
            print("[demo] no checkpoint found — running with random head")
        model.to(_device).eval()
        _model = model
    return _model, _device


def _disclaimer() -> str:
    _get_model()  # triggers _trained
    return DISCLAIMER_TRAINED if _trained else DISCLAIMER_UNTRAINED


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
        gr.Markdown(_disclaimer())
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
