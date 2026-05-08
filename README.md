# ScienceQA Visual Challenge — SmolVLM-500M-Instruct + QLoRA

End-to-end fine-tuning and inference pipeline for the *Pixels to Predictions* (ScienceQA) Kaggle competition. Final test score: **0.84507** (with TTA).

---

## 1. What's in this directory

| File | Purpose |
|---|---|
| [training.ipynb](training.ipynb) | Final training notebook (v10). Trains a LoRA adapter on `train + val` for 3 epochs and saves it to `./lora-final-seed42/`. |
| [inference.ipynb](inference.ipynb) | Inference notebook. Loads the saved adapter, runs log-likelihood scoring with Test-Time Augmentation, writes `submission.csv`. |
| [requirements.txt](requirements.txt) | Pinned package versions used in both notebooks. |
| [REPORT.md](REPORT.md) | Iteration-by-iteration writeup of all 10 versions tried. |
| [versions/](versions/) | Snapshots of every prior version of the training notebook. |

---

## 2. Environment

Both notebooks were developed and run on **Kaggle's free GPU runtime** (`Tesla T4`, 15.6 GB VRAM, fp16 only). The full pinned environment is in [requirements.txt](requirements.txt). The first cell of each notebook installs everything you need — you do not need to install anything manually before opening the notebook in Kaggle.

Key constraints driven by the T4 hardware:
- `fp16=True`, `bf16=False` (T4 has no bf16)
- No Flash Attention 2 (Ampere+ only)
- Gradient checkpointing on, BATCH_SIZE=1 (with GRAD_ACCUM=16 for effective batch 16)

If you replicate outside Kaggle, the same `requirements.txt` should work on any 16 GB+ NVIDIA GPU. On bf16-capable GPUs (A10/A100/H100/RTX 30+), changing `fp16=True` to `bf16=True` will give a small speedup.

---

## 3. Data setup on Kaggle

Both notebooks expect the competition dataset attached as a Kaggle Input dataset. The expected layout under `/kaggle/input/datasets/<owner>/finalexamdataset/` is:

```
finalexamdataset/
├── train.csv
├── val.csv
├── test.csv
├── sample_submission.csv
└── images/
    └── images/
        ├── train/
        ├── val/
        └── test/
```

The notebooks set:
```python
DATA_DIR   = Path("/kaggle/input/datasets/<owner>/finalexamdataset")
IMAGE_ROOT = DATA_DIR / "images"   # the inference notebook auto-detects this
```

If your Kaggle username is different from `komvopoulos`, update the `DATA_DIR` line in `cell-imports` (training) and `cell-inf-imports` (inference) accordingly.

---

## 4. Training — step by step

### 4.1 Open the notebook on Kaggle

1. Create a new Kaggle Notebook.
2. **Settings → Accelerator → GPU T4 ×1** (or T4 ×2; we pin to a single GPU regardless).
3. **Settings → Add data**: attach the competition dataset as a Kaggle dataset (the same one whose path matches `DATA_DIR` above).
4. Upload `training.ipynb` (File → Import → Notebook).

### 4.2 Run cells top to bottom

The notebook is structured for `Run All` to work. The key cells to watch for diagnostic output:

| Cell | What to verify |
|---|---|
| `cell-imports` | `Device: cuda`, `GPU: Tesla T4`, `VRAM: 15.6 GB` |
| `cell-load-csv` | `Train: 3,109 \| Val: 1,048 \| Test: 1,008` (counts approximate) |
| `cell-load-model` | `Trainable params : 4,784,128 / 512,266,432 (0.934%)`, `image_size : {'longest_edge': 1024}`, `image_splitting : True` |
| `cell 693d4ab3` (chat-template verification) | All 8 format checks should print `[OK ]` |
| `cell 4caa41c5` (sanity) | CHECK A: all 4 sample answer-letter positions correctly identified. CHECK B: `initial loss = 2-5` (NOT 100+) |
| `cell b8c60242` (smoke test) | 50-sample × 1-epoch run completes, loss decreasing |
| `cell-final-train` | `=== FINAL TRAINING START (fresh run) ===` then 770+ optimizer steps |

### 4.3 What to expect during final training

- Total steps: **777** (3 epochs × ~259 steps/epoch on `train + val`)
- Wall-clock time: **~4-5 hours** on T4
- Loss trajectory (should resemble v10's):
  - Step 10: ~65
  - Step 50: ~14
  - Step 100: ~12
  - Step 300: ~8
  - Step 500: ~7
  - Step 700+: ~5-6

If you ever see loss explode to ~110, something is misconfigured (most likely `label_smoothing_factor` got re-enabled — never enable it for Idefics3).

### 4.4 What gets saved

At the end of training the notebook calls:
```python
model.save_pretrained("./lora-final-seed42")
processor.save_pretrained("./lora-final-seed42")
```

This writes a ~30 MB directory containing `adapter_model.safetensors` and `adapter_config.json` (plus tokenizer files). Checkpoints are also saved every 50 steps under `./qlora-final/checkpoint-N/` (last 3 kept), used for cross-session resume if Kaggle times out.

### 4.5 Resuming after a Kaggle timeout

If your session expires before training finishes:
1. Download the latest `./qlora-final/checkpoint-N/` from the notebook's `Output` tab.
2. Re-upload it as a new Kaggle dataset (e.g. `komvopoulos/qlora-final-resume`).
3. In a fresh notebook session, set:
   ```python
   RESUME_CHECKPOINT_PATH = "/kaggle/input/qlora-final-resume/checkpoint-N"
   ```
4. Run all cells. The training will resume from that checkpoint.

The notebook also auto-detects a local `./qlora-final/` from a previous run in the same session — useful when re-running cells.

### 4.6 Saving the adapter for inference

After training completes:
1. **Option A (same Kaggle session)** — Run inference in the same kernel; `ADAPTER_DIR = "./lora-final-seed42"` works as-is.
2. **Option B (separate session)** —
   - Wait for the training notebook to finish and the kernel to commit.
   - Click **Output** in the right sidebar → download `lora-final-seed42` (or zip it via the optional download cell at the end of training).
   - Create a Kaggle dataset from the folder.
   - Attach that dataset to your inference notebook and set `ADAPTER_DIR` to its path.

---

## 5. Inference — step by step

### 5.1 Open and configure

1. Create a new Kaggle Notebook (or reuse the training session).
2. **Settings → Accelerator → GPU T4 ×1**.
3. **Settings → Add data**: attach the competition dataset (same as for training) AND, if running in a fresh session, your trained-adapter dataset.
4. Upload `inference.ipynb`.

### 5.2 Set the adapter path

In `cell-inf-imports`, set:

```python
ADAPTER_DIR = "./lora-final-seed42"   # same session

# or, in a fresh session with the adapter attached as a Kaggle dataset:
# ADAPTER_DIR = "/kaggle/input/<adapter-dataset-slug>/lora-final-seed42"
```

### 5.3 Run cells top to bottom

| Cell | What to verify |
|---|---|
| `cell-inf-load-model` | `image_size : {'longest_edge': 1024}`, `image_splitting : True`, `Adapter loaded from ...` |
| `2dbe72aa` (verification) | All 6 format checks print `[OK ]` |
| `cell-inf-score` | Defines TTA scorer with `USE_TTA = True` |
| `cell-inf-submission` | Pre-flight diagnostic shows varied log-probs (NOT all identical), then runs full test set inference |

### 5.4 TTA toggle

In `cell-inf-score`:
```python
USE_TTA = True   # default — averages original + reversed choice orderings
```

Set to `False` to recover single-pass scoring. Empirically TTA delivers **+2.29 % on val** for this adapter, so leave it on for submissions.

### 5.5 Time and output

- Test set: 1008 questions
- Time with TTA: **~75-100 minutes** on T4
- The notebook writes `submission.csv` to the working directory and triggers a browser download via the last cell.
- The working directory's `submission.csv` is also visible in the notebook's `Output` tab for manual download.

---

## 6. Submitting to Kaggle

1. Verify the `submission.csv` schema matches `sample_submission.csv` — two columns: `id`, `answer` (integer 0-9).
2. From the competition's *Submit Predictions* page, upload `submission.csv`.
3. Kaggle takes the **best** score across all your submissions for the leaderboard, so submitting iteratively is safe.

Expected score with the current configuration: **~0.84-0.85**.

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OSError: ... preprocessor_config.json` when loading adapter | Kaggle dataset upload dropped the processor files | Load processor from `MODEL_ID` instead of `ADAPTER_DIR` (already done in our `cell-inf-load-model`) |
| Sanity-check loss = 100+ instead of 2-5 | `label_smoothing_factor` accidentally enabled | Remove it from `TrainingArguments` — Idefics3 mis-aligns labels under HF's LabelSmoother |
| Sanity-check labels at wrong position / decoded letter mismatch | `collate_fn` walk-back skip set wrong | Verify `skip_ids = {eos_id, pad_id, im_end_id}` |
| Training loss collapses to <0.1 in epoch 1 | Overfitting (likely on a too-narrow LoRA target set with high LR) | Broaden `target_modules` to include MLP (`gate_proj`, `up_proj`, `down_proj`); reduce LR to 1e-4 |
| Format-check `[FAIL] Uses 'User:' (capitalized, inline)` | Custom ChatML format leaked back in | Ensure `build_prompt` calls `processor.apply_chat_template(...)` |
| OOM during smoke test or training | Image too large at `IMG_SIZE=1024` for unusual sample | Reduce `IMG_SIZE` to 768 (still 2× more visual info than the 384 baseline) |
| Inference loss-probs all identical in pre-flight | Walk-back is landing on a constant token | Re-check the chat-template verification cell — usually a prompt-format mismatch |

---

## 8. Reproducing the v10 result

The exact configuration that produced 0.84507:

```python
MODEL_ID            = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMG_SIZE            = 1024
processor.image_processor.size = {"longest_edge": 1024}
do_image_splitting  = True

LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]   # 4.78 M trainable params

NUM_EPOCHS          = 3
BATCH_SIZE          = 1
GRAD_ACCUM          = 16          # effective batch = 16
LR                  = 1e-4
lr_scheduler_type   = "cosine"
warmup_ratio        = 0.1
weight_decay        = 0.01
max_grad_norm       = 1.0
fp16                = True

TRAIN_WITH_SOLUTION = False
LECTURE_MAX_CHARS   = 1000
SEED                = 42
USE_TTA             = True   # at inference
```

Both `training.ipynb` and `inference.ipynb` already use this configuration as-is.

See [REPORT.md](REPORT.md) for the full iteration history of how this configuration was reached.
