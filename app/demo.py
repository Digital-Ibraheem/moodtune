"""Gradio demo for MoodTune. Loads checkpoints/best.pt if present."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import is_trained, predict_array  # noqa: E402

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


def predict(audio) -> dict:
    if audio is None:
        raise gr.Error("please provide audio via upload or microphone")
    sr, wav = audio
    return predict_array(sr, wav)


def build_app() -> gr.Blocks:
    disclaimer = DISCLAIMER_TRAINED if is_trained() else DISCLAIMER_UNTRAINED
    with gr.Blocks(title="MoodTune") as app:
        gr.Markdown(disclaimer)
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
