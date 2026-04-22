# MoodTune — Speech Emotion Recognition, with an immersive demo

<img width="1495" height="830" alt="image" src="https://github.com/user-attachments/assets/d51cb70b-27e4-4980-8c6a-875179c6af7c" />
<img width="1482" height="816" alt="image" src="https://github.com/user-attachments/assets/dce4a3ae-4203-4838-877a-4a72524c926f" />

A 4-class speech-emotion classifier (neutral / happy / sad / angry) with an honest cross-corpus evaluation and a voice-reactive browser UI. Trained from scratch on RAVDESS + CREMA-D, then fine-tuned on top of a stronger MSP-Podcast backbone for real-world mic robustness.

## The custom model

Everything in `src/` is ours. The interesting engineering decisions:

### Speaker-stratified splits
Random splits on RAVDESS leak speaker identity into the test set and inflate accuracy by 15–25 points. We split by **actor ID** — no voice appears in both train and test. 18 train / 2 val / 4 test on RAVDESS; speaker-held-out on CREMA-D too.

### Cross-corpus evaluation
Training on one corpus and testing on another is the only way to see if a model learned *emotion* or just *this dataset*. We report both numbers always.

### Three training regimes

10 epochs each on Apple Silicon MPS. `results/metrics.json` has the raw numbers.

| Regime                                    | RAVDESS test         | CREMA-D test        | Cross-corpus gap |
|-------------------------------------------|----------------------|---------------------|------------------|
| Random head (baseline)                    | 29.9%                | 26.3%               | —                |
| Head-only fine-tune, RAVDESS-only train   | **74.3%**            | 32.1%               | 42 pts           |
| Full fine-tune, RAVDESS-only train        | 70.8%                | 51.7%               | 19 pts           |
| **Head-only fine-tune, co-trained**       | 59.0%                | 55.7%               | **3 pts**        |

**The co-training result is the story.** Mixing RAVDESS + CREMA-D in the training set gives up absolute RAVDESS accuracy but collapses the cross-corpus gap from 42 points to 3. The per-class recall distribution also flattens — no more "everything is angry." This is what you'd want for a model that has to work on arbitrary speech.

### Additional training tricks added along the way
- **Silence trim + RMS normalize** at both train and serve time (consistent distribution).
- **Label smoothing 0.1** to tame overconfident softmax collapse.
- **Full fine-tune** option (`--unfreeze-encoder`) — also trains the transformer blocks.

![RAVDESS confusion](results/ravdess_confusion.png)
![CREMA-D confusion](results/crema_d_confusion.png)

## Serving a stronger backbone

Our 59/55 model still struggles on laptop-mic audio — both training corpora are *acted* studio speech. So the live demo serves a fine-tune of **[audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim](https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim)**, which is pretrained on 100+ hours of real podcast speech (MSP-Podcast). We fine-tune its classification head on our RAVDESS + CREMA-D labels — keeping the natural-speech feature quality while adapting to our 4-class label set.

The dimensional outputs (arousal / valence) map to 4 classes via a tiny circumplex projection in `src/audeering_inference.py`. The backend picks a backbone via `MOODTUNE_BACKEND=local|superb|audeering`.

## Web UI

A voice-reactive Vite + React frontend (`frontend/`). Dark, minimal, cinematic:
- WebGL fragment shader that quivers with your live voice amplitude + FFT bands.
- 4-second tap-to-record with a countdown ring and live amplitude meter.
- Result: huge fade-in of the predicted emotion, background tints toward that emotion's color.
- Client-side decode + 16 kHz WAV encode so the backend needs no ffmpeg.

## Running it

```bash
# One-time
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m data.download && python -m data.prepare     # RAVDESS zip + CREMA-D tarball

# Train (optional — skip and use audeering out of the box)
python -m src.train --epochs 10                        # head-only, co-trained
python -m src.train --epochs 10 --unfreeze-encoder --lr 3e-5  # full fine-tune
python -m src.evaluate                                 # writes results/

# Serve
uvicorn app.server:app --port 8000                     # backend (defaults to audeering)
cd frontend && npm install && npm run dev              # frontend on :5173
```

## Roadmap

- [x] Speaker-stratified splits
- [x] Head-only + full fine-tune
- [x] Co-training to close the cross-corpus gap
- [x] Silence trim + RMS normalize + label smoothing
- [x] MSP-Podcast backbone with a fine-tuned head
- [x] Voice-reactive web UI
- [ ] SpecAugment / noise augmentation for mic robustness
- [ ] Domain-adversarial training
- [ ] Multilingual via XLSR-53
- [ ] Calibration plot

## Limitations
- Training data is entirely acted. Real conversational emotion is messier.
- English only.
- Fixed 4-second windows.
- Audeering's weights are CC-BY-NC-SA — non-commercial only. Our own weights are MIT.

## References
- Baevski et al. (2020). *wav2vec 2.0.* NeurIPS.
- Livingstone & Russo (2018). *RAVDESS.* PLoS ONE.
- Cao et al. (2014). *CREMA-D.* IEEE TAFFC.
- Wagner et al. (2023). *Dawn of the transformer era in speech emotion recognition.* (audeering / MSP-Podcast)

BibTeX in `CITATIONS.bib`. Code: MIT.
