"""Fine-tune Wav2Vec2 on RAVDESS.

Default recipe: freeze the encoder (conv feature extractor + transformer blocks),
train only the projector + linear classifier. This is the fastest route to a
usable model and what the README's "frozen encoder" column reports.

Pass --unfreeze-encoder to also train the transformer blocks (slower, usually
better). The conv feature extractor stays frozen either way, per the Wav2Vec2
fine-tuning convention.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHECKPOINT_DIR, LABELS, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from src.dataset import EmotionAudioDataset, make_dataloader  # noqa: E402
from src.model import build_model, count_parameters  # noqa: E402


def _freeze_head_only(model) -> None:
    """Freeze everything except projector + classifier."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("projector") or name.startswith("classifier")


def _evaluate(model, loader, device) -> dict:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    losses: list[float] = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            inputs = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            logits = model(input_values=inputs).logits
            losses.append(loss_fn(logits, y).item())
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)) if labels else 0.0,
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)) if labels else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "best.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = PROCESSED_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found at {manifest_path}. Run data.prepare first.")
    manifest = pd.read_csv(manifest_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}  (unfreeze_encoder={args.unfreeze_encoder})")

    model, _ = build_model(num_labels=len(LABELS))
    if args.unfreeze_encoder:
        for p in model.parameters():
            p.requires_grad = True
        model.freeze_feature_encoder()
    else:
        _freeze_head_only(model)
    count_parameters(model)
    model.to(device)

    train_loader = make_dataloader(
        EmotionAudioDataset(manifest, split="train"), batch_size=args.batch_size, shuffle=True
    )
    val_loader = make_dataloader(
        EmotionAudioDataset(manifest, split="val"), batch_size=args.batch_size, shuffle=False
    )
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    history: list[dict] = []
    best_val_acc = -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}")
        for batch in pbar:
            inputs = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            optimizer.zero_grad()
            logits = model(input_values=inputs).logits
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(epoch_losses[-20:]):.4f}")

        train_loss = float(np.mean(epoch_losses))
        val = _evaluate(model, val_loader, device)
        wall = time.time() - t0
        print(
            f"  epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
            f"val_f1={val['macro_f1']:.4f}  ({wall:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "wall_seconds": wall,
            }
        )

        if val["accuracy"] > best_val_acc:
            best_val_acc = val["accuracy"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": val["accuracy"],
                    "val_macro_f1": val["macro_f1"],
                    "unfreeze_encoder": args.unfreeze_encoder,
                },
                args.checkpoint,
            )
            print(f"    ↳ saved {args.checkpoint} (val_acc={val['accuracy']:.4f})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = RESULTS_DIR / "training_history.json"
    hist_path.write_text(json.dumps({"config": vars(args) | {"checkpoint": str(args.checkpoint)}, "history": history}, indent=2, default=str))
    print(f"\nbest val accuracy: {best_val_acc:.4f}  (checkpoint: {args.checkpoint})")
    print(f"wrote {hist_path}")


if __name__ == "__main__":
    main()
