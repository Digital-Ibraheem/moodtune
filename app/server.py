"""FastAPI inference server for the MoodTune web frontend.

Thin HTTP layer over the predict function. Accepts a WAV blob (frontend
normalizes to 16kHz mono before POSTing) and returns 4-class probabilities.

MOODTUNE_BACKEND env var picks the model:
  - "audeering" (default): audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
                           — trained on MSP-Podcast natural speech, best for mic use.
  - "superb"             : superb/wav2vec2-base-superb-er — IEMOCAP-trained,
                           lightweight fallback.
  - "local"              : our own fine-tuned RAVDESS+CREMA-D head at
                           checkpoints/best.pt. Keeps the custom-training story alive.
"""

from __future__ import annotations

import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BACKEND = os.environ.get("MOODTUNE_BACKEND", "audeering").lower()
if BACKEND == "local":
    from src.inference import get_model, predict_array  # noqa: E402,F401
elif BACKEND == "superb":
    from src.superb_inference import get_model, predict_array  # noqa: E402,F401
else:
    from src.audeering_inference import get_model, predict_array  # noqa: E402,F401

SILENCE_THRESHOLD = 0.01


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the model once so the first real request doesn't pay the load cost.
    get_model()
    yield


app = FastAPI(title="MoodTune", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    result = get_model()
    device = result[-1] if isinstance(result, tuple) else None
    return {"ok": True, "backend": BACKEND, "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")

    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not decode audio: {exc}")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < SILENCE_THRESHOLD:
        return {"error": "silence", "peak": peak}

    probs = predict_array(sr, wav)
    top_label = max(probs, key=probs.get)
    return {"probs": probs, "top": top_label, "peak": peak, "duration": len(wav) / sr}
