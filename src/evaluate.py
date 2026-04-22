"""Zero-shot evaluation of Wav2Vec2 with an untrained classifier head.

Produces confusion matrices and a metrics.json under results/. All numbers
are expected to be near-chance; this is the baseline before any training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHECKPOINT_DIR, LABELS, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from src.dataset import EmotionAudioDataset, make_dataloader  # noqa: E402
from src.model import build_model  # noqa: E402


def evaluate_split(model, feature_extractor, manifest, split, device, batch_size=8, corpus=None, desc_suffix=""):
    subset = manifest[manifest["split"] == split]
    if corpus is not None:
        subset = subset[subset["corpus"] == corpus]
    if subset.empty:
        return None
    dataset = EmotionAudioDataset(subset)
    loader = make_dataloader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    desc = f"eval[{corpus or 'all'}:{split}{desc_suffix}]"
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            inputs = batch["input_values"].to(device)
            logits = model(input_values=inputs).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].tolist())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels_arr, preds_arr, labels=list(range(len(LABELS))), zero_division=0
    )

    return {
        "split": split,
        "n": int(len(labels_arr)),
        "accuracy": float(accuracy_score(labels_arr, preds_arr)),
        "macro_f1": float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)),
        "per_class_precision": {lbl: float(precision[i]) for i, lbl in enumerate(LABELS)},
        "per_class_recall": {lbl: float(recall[i]) for i, lbl in enumerate(LABELS)},
        "predictions": preds_arr.tolist(),
        "labels": labels_arr.tolist(),
    }


def plot_confusion_matrix(labels, preds, label_names, save_path: Path, title: str) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(label_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar=False,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _format_table(res: dict) -> str:
    if res is None:
        return "(no data)"
    lines = [
        f"  n        : {res['n']}",
        f"  accuracy : {res['accuracy']:.4f}",
        f"  macro_f1 : {res['macro_f1']:.4f}",
        "  per-class precision / recall:",
    ]
    for lbl in LABELS:
        p = res["per_class_precision"][lbl]
        r = res["per_class_recall"][lbl]
        lines.append(f"    {lbl:<8} P={p:.3f} R={r:.3f}")
    return "\n".join(lines)


def main() -> None:
    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(
            f"manifest not found at {manifest_path}. "
            "Run `python -m data.download && python -m data.prepare` first."
        )
    manifest = pd.read_csv(manifest_path)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")
    model, fe = build_model(num_labels=len(LABELS))

    checkpoint_path = CHECKPOINT_DIR / "best.pt"
    model_state = "untrained_classifier_head"
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model_state = (
            "trained_full_finetune" if ckpt.get("unfreeze_encoder")
            else "trained_head_only"
        )
        print(f"loaded {checkpoint_path}  ({model_state}, val_acc={ckpt.get('val_accuracy', 'n/a')})")
    model.to(device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics: dict = {"model_state": model_state, "splits": {}}

    rav_test = evaluate_split(model, fe, manifest, "test", device, corpus="ravdess")
    if rav_test is not None:
        plot_confusion_matrix(
            rav_test["labels"],
            rav_test["predictions"],
            LABELS,
            RESULTS_DIR / "ravdess_confusion.png",
            f"RAVDESS test — {model_state}",
        )
        metrics["splits"]["ravdess_test"] = {k: v for k, v in rav_test.items() if k not in {"predictions", "labels"}}
        print("\n[RAVDESS test — 4 held-out speakers]")
        print(_format_table(rav_test))

    # Accept either the new co-train split (corpus=crema_d, split=test) or the
    # legacy manifest where CREMA-D was entirely split=ood.
    crema = evaluate_split(model, fe, manifest, "test", device, corpus="crema_d")
    if crema is None:
        crema = evaluate_split(model, fe, manifest, "ood", device, corpus="crema_d")
        crema_key = "crema_d_ood"
        crema_title_split = "OOD"
    else:
        crema_key = "crema_d_test"
        crema_title_split = "test (15 held-out speakers)"
    if crema is not None:
        plot_confusion_matrix(
            crema["labels"],
            crema["predictions"],
            LABELS,
            RESULTS_DIR / "crema_d_confusion.png",
            f"CREMA-D {crema_title_split} — {model_state}",
        )
        metrics["splits"][crema_key] = {k: v for k, v in crema.items() if k not in {"predictions", "labels"}}
        print(f"\n[CREMA-D {crema_title_split}]")
        print(_format_table(crema))

    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
