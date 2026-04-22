"""Paths, labels, and model constants shared across the project."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"

SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 4.0
MAX_SAMPLES = int(SAMPLE_RATE * MAX_DURATION_SECONDS)

LABELS = ["neutral", "happy", "sad", "angry"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

MODEL_NAME = "facebook/wav2vec2-base"

# RAVDESS: 3rd filename segment encodes emotion.
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
# Calm is merged into neutral; fearful / disgust / surprised are dropped.
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# CREMA-D: 3-letter emotion codes. NEU HAP SAD ANG FEA DIS.
CREMA_EMOTION_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
}
