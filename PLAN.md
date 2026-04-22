# MoodTune — Build Plan

## What this is

A speech emotion recognition portfolio project. Fine-tunes Wav2Vec2 on RAVDESS for 4-class emotion classification (happy, sad, angry, neutral), evaluates cross-corpus generalization on CREMA-D, and ships a Gradio demo.

**Training is out of scope for this build.** Use the pretrained Wav2Vec2 backbone and a randomly-initialized classifier head. All evaluation numbers in the README should be clearly labeled as "untrained head baseline" — this is a work-in-progress portfolio project, and reporting honest zero-shot numbers is part of the methodology story.

The point of this project is **methodology, framing, and presentation**, not performance. The README, repo structure, and Gradio demo are the deliverables. Training can happen later on a GPU machine — this build sets up everything to make training a one-command operation.

## Build philosophy

- Ship a complete, runnable project. Every step should leave the repo in a working state.
- Commit after each logical step with clear conventional-commit messages.
- **Do not add "Co-Authored-By: Claude" or any Claude/Anthropic attribution to commit messages.** Commit as if the user wrote the code.
- Write minimal, readable code. No over-engineering, no unnecessary abstractions.
- If something is ambiguous, pick a reasonable default and note it in the README rather than stopping to ask.

## Repo structure

```
moodtune/
├── README.md
├── PLAN.md                   (this file)
├── requirements.txt
├── .gitignore
├── LICENSE                   (MIT)
├── data/
│   ├── __init__.py
│   ├── download.py           # pulls RAVDESS + CREMA-D
│   └── prepare.py            # splits, label mapping, manifest generation
├── src/
│   ├── __init__.py
│   ├── dataset.py            # PyTorch Dataset, audio loading, resampling
│   ├── model.py              # Wav2Vec2 + classification head
│   ├── evaluate.py           # in-corpus + cross-corpus eval, confusion matrix
│   └── config.py             # hyperparameters, paths
├── app/
│   ├── __init__.py
│   └── demo.py               # Gradio app
├── results/
│   └── .gitkeep              # metrics + confusion matrices go here
└── checkpoints/
    └── .gitkeep              # trained weights go here, gitignored
```

## Step-by-step build

Each step ends with a commit. Commit messages use conventional commits format (`feat:`, `chore:`, `docs:`, etc.). Do NOT include any Claude/Anthropic/AI attribution in commit messages, bodies, or trailers.

### Step 0 — Repo scaffold

- Initialize git repo
- Create directory structure above
- Add `.gitignore` covering: `__pycache__/`, `*.pyc`, `.venv/`, `venv/`, `.env`, `data/raw/`, `data/processed/`, `checkpoints/*.pt`, `checkpoints/*.bin`, `.DS_Store`, `.ipynb_checkpoints/`, `*.egg-info/`, `results/*.png` (keep `.gitkeep`)
- Add MIT `LICENSE`
- Add `requirements.txt` with pinned-minor versions:
  - `torch`
  - `torchaudio`
  - `transformers`
  - `datasets`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `gradio`
  - `numpy`
  - `pandas`
  - `tqdm`
  - `soundfile`
  - `librosa`

**Commit:** `chore: initial repo scaffold`

### Step 1 — Config module

Create `src/config.py` with:

- `DATA_DIR`, `RAW_DIR`, `PROCESSED_DIR`, `CHECKPOINT_DIR`, `RESULTS_DIR` path constants (use `pathlib.Path`)
- `SAMPLE_RATE = 16000`
- `MAX_DURATION_SECONDS = 4.0`
- `MAX_SAMPLES = SAMPLE_RATE * MAX_DURATION_SECONDS`
- `LABELS = ["neutral", "happy", "sad", "angry"]`
- `LABEL_TO_ID`, `ID_TO_LABEL` dicts
- `MODEL_NAME = "facebook/wav2vec2-base"`
- `RAVDESS_EMOTION_MAP` — RAVDESS encodes emotion as the 3rd filename segment: `01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised`. Map `01,02 → neutral`, `03 → happy`, `04 → sad`, `05 → angry`. Others → drop.
- `CREMA_EMOTION_MAP` — CREMA-D uses 3-letter codes: `NEU, HAP, SAD, ANG, FEA, DIS`. Map `NEU → neutral`, `HAP → happy`, `SAD → sad`, `ANG → angry`. Others → drop.

**Commit:** `feat: add config module with label maps and paths`

### Step 2 — Data download

Create `data/download.py`:

- Function `download_ravdess(target_dir)`:
  - Uses `huggingface_hub` or `datasets` to pull `narad/ravdess` (or falls back to direct Zenodo download of `Audio_Speech_Actors_01-24.zip`)
  - Extracts to `data/raw/ravdess/`
  - Skips if already present
- Function `download_crema_d(target_dir)`:
  - Clones or downloads from the CREMA-D GitHub repo (`CheyneyComputerScience/CREMA-D`) — the `AudioWAV/` folder
  - If the full clone is too heavy, use GitHub's archive download for just the AudioWAV directory
  - Extracts to `data/raw/crema_d/`
  - Skips if already present
- `main()` runs both, with `--ravdess-only` and `--crema-only` flags
- Include a docstring header explaining the licensing: RAVDESS is CC BY-NA-SC 4.0, CREMA-D is Open Database License

**Note on size:** CREMA-D is ~500MB. If the download is fragile, fall back to a sampling mode that grabs only ~200 files for the demo. Log which mode was used.

**Commit:** `feat: add dataset download scripts for RAVDESS and CREMA-D`

### Step 3 — Data preparation

Create `data/prepare.py`:

- Function `parse_ravdess_filename(path)` → returns `{emotion, actor_id, intensity, statement, repetition}` parsed from the RAVDESS naming convention
- Function `build_ravdess_manifest(raw_dir)`:
  - Walks `data/raw/ravdess/`, parses every `.wav` file
  - Applies `RAVDESS_EMOTION_MAP`, drops unmapped emotions
  - Computes duration via `soundfile.info()` (fast, doesn't load audio)
  - Returns DataFrame with columns: `filepath, label, label_id, speaker_id, duration, corpus`
- Function `build_crema_d_manifest(raw_dir)`:
  - Same pattern. CREMA-D filenames: `ActorID_Sentence_Emotion_Intensity.wav`
  - Speaker ID is the first segment
- Function `speaker_stratified_split(df, val_speakers=2, test_speakers=4, seed=42)`:
  - Splits RAVDESS by actor ID — critical methodology point
  - RAVDESS has 24 actors; hold out 4 for test, 2 for val, rest for train
  - Returns df with new `split` column
- `main()`:
  - Builds both manifests
  - Applies speaker-stratified split to RAVDESS (`split ∈ {train, val, test}`)
  - Labels CREMA-D entirely as `split=ood` (out-of-distribution)
  - Saves to `data/processed/manifest.csv`
  - Prints summary: total samples per corpus, class balance, speaker counts

**Commit:** `feat: add manifest builder with speaker-stratified splits`

### Step 4 — PyTorch Dataset

Create `src/dataset.py`:

- `EmotionAudioDataset(Dataset)`:
  - Takes manifest DataFrame + optional split filter
  - `__getitem__` loads WAV via `torchaudio.load`, resamples to 16kHz if needed (cache a resampler per source sample rate), pads or truncates to `MAX_SAMPLES`
  - Returns `{input_values: Tensor, label: int, speaker_id: str, corpus: str}`
  - Input is raw waveform, not spectrogram — Wav2Vec2 handles feature extraction internally
- `make_dataloader(dataset, batch_size, shuffle)` helper with a simple collator

Add a tiny `__main__` smoke test that loads the manifest, instantiates the dataset for the `train` split, grabs one batch, and prints tensor shapes. This verifies the pipeline end-to-end without training.

**Commit:** `feat: add PyTorch Dataset for emotion audio`

### Step 5 — Model

Create `src/model.py`:

- `build_model(num_labels=4, freeze_feature_extractor=True)`:
  - Loads `Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)`
  - Freezes the convolutional feature extractor (`model.freeze_feature_encoder()`)
  - Returns model + the matching `Wav2Vec2FeatureExtractor`
- `count_parameters(model)` helper that prints trainable vs. frozen param counts
- `__main__` smoke test: builds model, prints parameter summary, runs a forward pass on a dummy 4-second random waveform, prints output logits shape. No training, no gradients.

**Commit:** `feat: add Wav2Vec2 model builder`

### Step 6 — Evaluation (zero-shot untrained head)

Create `src/evaluate.py`:

- `evaluate_split(model, feature_extractor, manifest, split, device, batch_size=8)`:
  - Runs inference on the specified split
  - Returns dict with `accuracy`, `macro_f1`, `per_class_precision`, `per_class_recall`, `predictions`, `labels`
- `plot_confusion_matrix(labels, preds, label_names, save_path, title)`:
  - Uses seaborn heatmap
  - Saves PNG to `results/`
- `main()`:
  - Loads manifest, builds model (untrained classifier head)
  - Evaluates on RAVDESS test split → saves `results/ravdess_confusion.png` + metrics
  - Evaluates on CREMA-D (full OOD split) → saves `results/crema_d_confusion.png` + metrics
  - Writes all numbers to `results/metrics.json` with a top-level note: `"model_state": "untrained_classifier_head"`
  - Prints a formatted summary table to stdout

**Important:** All of this runs on CPU in reasonable time because it's inference-only on a small dataset. No GPU required. Use `batch_size=8` and `torch.no_grad()`.

This step generates real numbers for the README. They will be near-chance (~25% for 4-class uniform), and that's fine — the README frames this as the starting baseline.

**Commit:** `feat: add evaluation with confusion matrices for in-corpus and cross-corpus`

### Step 7 — Gradio demo

Create `app/demo.py`:

- Gradio Blocks app with:
  - Title: "MoodTune — Speech Emotion Recognition"
  - Subtitle markdown with a prominent disclaimer banner: "⚠️ This is an in-progress portfolio project. The classifier head is currently untrained — predictions are not meaningful yet. Training pending on a GPU machine. See README for methodology."
  - Audio input component (supports both microphone recording and file upload)
  - "Analyze" button
  - Output: bar chart (matplotlib or Gradio's `Label` component with `num_top_classes=4`) showing probability per emotion
  - Small "About" accordion explaining the project and linking to the GitHub repo
- `predict(audio)`:
  - Loads audio, resamples to 16kHz, pads/truncates to 4s
  - Runs through the model, softmaxes logits
  - Returns dict `{label: probability}` for Gradio's Label component
- `main()` launches with `share=False` by default, configurable via `--share` flag

**Commit:** `feat: add Gradio demo with untrained-model disclaimer`

### Step 8 — README

**Write this last**, after Step 6 has produced real numbers. The README is the highest-leverage file. Structure:

1. **Title + one-line pitch**
   > MoodTune — Speech Emotion Recognition with honest cross-corpus evaluation.

2. **Status banner** — "🚧 Work in progress. Classifier head untrained; pipeline and evaluation methodology complete. Training pending."

3. **Demo screenshot placeholder** — leave a `![demo](docs/demo.png)` line with a TODO comment. User will add later.

4. **What this project is** — 3-4 sentences. What it does, why SER is hard, what makes this repo different from the generic RAVDESS tutorials.

5. **Key design decisions** — this section is the project's value prop. Each as a short paragraph:
   - **Why speaker-stratified splits.** Random splits on RAVDESS leak speaker identity between train/test, inflating accuracy. Splitting by actor ID (24 actors → 18 train / 2 val / 4 test) measures actual emotion generalization, not voice memorization.
   - **Why cross-corpus evaluation.** A model trained on RAVDESS is almost always deployed on non-RAVDESS audio. Evaluating on CREMA-D (different actors, different sentences, different recording conditions) surfaces the domain-shift gap that single-corpus benchmarks hide.
   - **Why 4 classes, not 8.** Humans disagree on "fearful" vs. "surprised" and "calm" vs. "neutral." Collapsing to {neutral, happy, sad, angry} improves inter-rater reliability and produces a cleaner evaluation signal.
   - **Why Wav2Vec2.** Self-supervised pretraining on LibriSpeech gives a strong audio representation for free. Fine-tuning a linear head is the standard 2024 approach and requires orders of magnitude less data than training from scratch.

6. **Current results** — two small tables (RAVDESS test, CREMA-D OOD) showing accuracy + macro-F1 from Step 6. Label them clearly as `untrained classifier head — baseline only`. Include the two confusion matrices as images.

7. **Roadmap** — bulleted list of what training will add:
   - Fine-tune classifier head on RAVDESS train split
   - Compare frozen-encoder vs. full fine-tuning
   - Report cross-corpus drop with trained model
   - Add SpecAugment and waveform augmentation
   - Extend to multilingual SER via XLSR

8. **Architecture** — brief explanation of the data pipeline + model, maybe a small diagram (ASCII or mermaid is fine).

9. **Running it locally** — clear commands:
   ```bash
   pip install -r requirements.txt
   python -m data.download
   python -m data.prepare
   python -m src.evaluate
   python -m app.demo
   ```

10. **Limitations** — honest list: acted emotional speech (RAVDESS actors), small speaker pool, English-only, no noise robustness, fixed 4-second windows, untrained head. Frame as known tradeoffs, not apologies.

11. **References** — Wav2Vec2 paper (Baevski et al. 2020), RAVDESS paper (Livingstone & Russo 2018), CREMA-D paper (Cao et al. 2014), plus a note on the BibTeX citations being in `CITATIONS.bib` if you add one.

12. **License** — MIT for code, note that RAVDESS and CREMA-D have their own licenses.

The README should read as if written by someone who thinks carefully about a hard problem. Not apologetic about the untrained state; framing it as a work-in-progress with a deliberate build order.

**Commit:** `docs: add README with methodology and current baseline results`

### Step 9 — Final polish

- Run everything end-to-end one more time to confirm no breakage
- Make sure `results/metrics.json` is committed
- Make sure confusion matrix PNGs are committed (they're tiny)
- Verify `.gitignore` is not accidentally ignoring them
- Add a `CITATIONS.bib` with BibTeX for the three key papers
- Quick sanity pass on README: no broken links, commands actually work from a fresh clone

**Commit:** `chore: final polish and citations`

## Commit message rules (read this carefully)

- Use conventional commits: `feat:`, `fix:`, `docs:`, `chore:`, `refactor:`, `test:`
- Keep subject lines under 72 characters
- **Do NOT include any of the following in any commit message, body, or trailer:**
  - `Co-Authored-By: Claude <...>`
  - `Generated with Claude Code`
  - `🤖 Generated with...`
  - Any mention of AI, Claude, Anthropic, LLM, or assistant involvement
- Write commits as if the user is the sole author. This is the user's portfolio project.

## What NOT to do

- Don't train the model. The point of this build is the scaffolding, methodology, and presentation.
- Don't tune hyperparameters. There are no hyperparameters to tune yet.
- Don't add a `train.py` skeleton beyond what's needed. Training is a future step.
- Don't add CI/CD, Docker, pre-commit hooks, or other infrastructure unless trivially small. Scope creep kills portfolio projects.
- Don't write unit tests. This is a portfolio project, not a library. A few `__main__` smoke tests in key modules is plenty.
- Don't add `wandb`, `mlflow`, `hydra`, or other heavy ML tooling. Keep the stack minimal.
- Don't upload datasets or checkpoints to git. `.gitignore` handles this.

## If you get stuck

- Dataset download failing → fall back to a smaller subset and note it in the README
- HuggingFace authentication issues → use anonymous access, RAVDESS and Wav2Vec2-base are both public
- CREMA-D too big to fully download → subsample to 500 files for the manifest and note it
- Model doesn't load → check the transformers version in `requirements.txt`, pin to a known-working version

Stop and ask the user only if: licensing looks ambiguous, a download source is gone, or a step would take dramatically longer than the plan suggests.

## Success criteria

When this is done, the user should be able to:

1. Clone the repo, `pip install -r requirements.txt`, run the 4 commands in the README, and see a working Gradio demo with the disclaimer banner
2. Open the README and see a clean story: what the project is, why the methodology choices matter, what the current numbers are, what's next
3. Put the repo link on a resume and have a recruiter understand what it does in under 30 seconds
4. Have an interview conversation about speaker-stratified splits and cross-corpus evaluation that sounds informed

Ship it.
