# MoodTune — Speech Emotion Recognition with honest cross-corpus evaluation

<img width="1600" height="1514" alt="screen" src="https://github.com/user-attachments/assets/8c822a53-f401-45a7-8d72-f3885d8934ac" />



## What this project is

MoodTune is a 4-class speech-emotion recognition pipeline (neutral / happy / sad / angry) built on top of a pretrained [Wav2Vec2](https://arxiv.org/abs/2006.11477) encoder. It trains on [RAVDESS](https://zenodo.org/record/1188976) and evaluates **both** in-corpus (RAVDESS held-out speakers) and **out-of-distribution** on [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) — a different set of actors, sentences, and recording conditions.

Most public RAVDESS tutorials report accuracy on a random 80/20 split and stop there. That's a weak benchmark: the same 24 actors appear in train and test, so models memorize voices more than they learn emotion. MoodTune is about taking the harder evaluation seriously.

## Key design decisions

### Why speaker-stratified splits
Random splits on RAVDESS leak speaker identity between train and test. The model learns which voice maps to which label rather than what emotion sounds like. MoodTune splits by actor ID — 24 total → **18 train / 2 val / 4 test** — so every voice at eval time is one the model has never heard. The resulting numbers are lower than the inflated benchmarks you see online, and that's the point.

### Why cross-corpus evaluation
A classifier trained on RAVDESS is almost always deployed on non-RAVDESS audio. Evaluating on CREMA-D — different actors, different sentences, different microphones, more varied intensity — surfaces the domain-shift gap that single-corpus benchmarks hide. This drop is usually the difference between a paper result and a demo that works in the wild.

### Why 4 classes, not 8
RAVDESS ships with 8 emotions. Inter-rater agreement on "fearful" vs. "surprised" and "calm" vs. "neutral" is poor — human listeners themselves disagree. Collapsing to {neutral, happy, sad, angry} merges calm into neutral, drops the three hardest classes, and produces a cleaner evaluation signal. The mapping is in `src/config.py`.

### Why Wav2Vec2
Self-supervised pretraining on LibriSpeech gives a strong audio representation for free. Fine-tuning a small classification head on a frozen encoder is the standard 2024 recipe and needs orders of magnitude less data than training from scratch. MoodTune uses `facebook/wav2vec2-base` with the feature encoder frozen; only the projection + classification layers are trainable.

## Current results

Two training regimes, 10 epochs each on Apple Silicon MPS. Raw numbers in [`results/metrics.json`](results/metrics.json):

| Model                          | RAVDESS test (in-corpus) | CREMA-D (OOD)   | Cross-corpus gap | Train time |
|--------------------------------|--------------------------|-----------------|------------------|------------|
| Random head (baseline)         | 29.9% / F1 0.19          | 26.3% / F1 0.11 | —                | —          |
| **Head-only fine-tune**        | **74.3% / F1 0.71**      | 32.1% / F1 0.23 | **42 pts**       | ~5 min     |
| **Full fine-tune (encoder unfrozen)** | 70.8% / F1 0.70   | **51.7% / F1 0.48** | **19 pts**   | ~10 min    |

**The interesting result is the trade-off.** Head-only fine-tune is slightly better on RAVDESS itself (the projector + classifier squeeze the most out of RAVDESS-specific embedding directions) but collapses on CREMA-D. Full fine-tuning gives up a few points on RAVDESS but nearly **doubles** cross-corpus accuracy — unfreezing the transformer blocks lets the model re-learn audio representations that are less tied to RAVDESS's recording conditions, 2 stock sentences, and theatrical intensity distribution.

A 19-point cross-corpus gap is still large. That's honest: closing it further needs real domain-adaptation techniques (augmentation, adversarial objectives, or training on a mix of corpora) — see the roadmap.

Most public RAVDESS notebooks post 85–95% accuracy on a random train/test split. Our 74% / 71% numbers come from a **speaker-stratified** split: no actor appears in both train and test. Those published numbers are inflated by speaker leakage.

![RAVDESS confusion](results/ravdess_confusion.png)
![CREMA-D confusion](results/crema_d_confusion.png)

Reproduce with:

```bash
python -m src.train --epochs 10                    # head-only
python -m src.train --epochs 10 --lr 3e-5 --unfreeze-encoder  # full fine-tune
python -m src.evaluate                             # uses checkpoints/best.pt
```

## Roadmap

- [x] Fine-tune classifier head on RAVDESS (frozen encoder)
- [x] Full fine-tune (unfreeze transformer)
- [x] Report the cross-corpus accuracy drop for both regimes
- [ ] Regularize the full fine-tune — label smoothing, dropout on the head, earlier stopping; val loss diverged by epoch 4
- [ ] Add SpecAugment and waveform-level augmentation (noise, pitch, time-stretch)
- [ ] Domain-adversarial training to close the remaining 19-point RAVDESS → CREMA-D gap
- [ ] Co-training on CREMA-D + RAVDESS and measuring transfer to IEMOCAP
- [ ] Extend to multilingual SER via XLSR-53
- [ ] Add a calibration plot — SER models are usually overconfident OOD

## Architecture

```
            ┌──────────────┐
  raw .wav  │  resample    │  waveform (16kHz, 4s, pad/truncate)
  ────────► │  to 16kHz    │ ───────────────────────────────────┐
            └──────────────┘                                     │
                                                                 ▼
                                       ┌───────────────────────────────┐
                                       │  Wav2Vec2 base (frozen enc.)  │
                                       └──────────────┬────────────────┘
                                                      │ hidden states
                                                      ▼
                                       ┌───────────────────────────────┐
                                       │  mean-pool + linear head      │  → 4 logits
                                       └───────────────────────────────┘
```

Data layer: `data/download.py` pulls RAVDESS (Zenodo zip) and CREMA-D (GitHub raw files). `data/prepare.py` parses filenames, maps to the 4-class label space, applies speaker-stratified splits to RAVDESS, and marks CREMA-D as `split=ood`. Result: a single `data/processed/manifest.csv`.

Model layer: `src/model.py` wraps `Wav2Vec2ForSequenceClassification` with the feature encoder frozen. `src/dataset.py` handles audio loading, resampling, and fixed-length padding. `src/evaluate.py` runs inference and emits metrics + confusion matrices.

## Running it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Pull datasets (RAVDESS ~200MB; CREMA-D is larger — use --sample for a fast demo)
python -m data.download --sample

# Build the manifest (splits, label mapping)
python -m data.prepare

# Fine-tune the head — ~5 min on Apple Silicon MPS, saves checkpoints/best.pt
python -m src.train --epochs 10

# Eval — auto-loads checkpoints/best.pt if present, writes metrics + PNGs
python -m src.evaluate

# Launch the Gradio demo
python -m app.demo
```

Inference and the Gradio demo run on CPU. Training uses MPS on Apple Silicon or CUDA if available; the `--unfreeze-encoder` flag trades ~10x longer training for a few points of accuracy.

## Limitations

- Acted emotional speech. RAVDESS and CREMA-D are recorded by voice actors reading fixed sentences. Real-world emotional speech is messier, shorter, and overlapping. SER models trained on acted corpora routinely degrade on spontaneous speech.
- Small speaker pool. 24 RAVDESS actors × 91 CREMA-D actors is enough for a demo, not for a production model.
- English-only. The encoder is trained on English (LibriSpeech). XLSR is in the roadmap.
- No noise robustness. Clean studio recordings only. No channel, compression, or background-noise augmentation yet.
- Fixed 4-second windows. Long utterances are truncated, short ones padded with silence — which the encoder can over-weight.
- Head-only fine-tune. Only the projector + classifier are trained; the Wav2Vec2 transformer is frozen. Unfreezing it (`--unfreeze-encoder`) should close some of the cross-corpus gap.

## References

- Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* NeurIPS.
- Livingstone, S. R., & Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).* PLoS ONE 13(5).
- Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Transactions on Affective Computing.

BibTeX entries are in `CITATIONS.bib`.

## License

Code is MIT (`LICENSE`). The RAVDESS corpus is CC BY-NC-SA 4.0; CREMA-D is released under the Open Database License. Neither is redistributed by this repo — the download scripts pull directly from the original sources.
