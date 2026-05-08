# Fine-Tuning SmolVLM-500M-Instruct for ScienceQA — Iteration Report

## 1. Setup

**Task.** Multiple-choice science QA with images. Each question has 2-10 textual choices and a reference image; the model must select the single correct letter.

**Model.** `HuggingFaceTB/SmolVLM-500M-Instruct` (required by competition rules). Idefics3 architecture: SigLIP-style vision encoder (≈93 M params, 512×512 patches → 64 visual tokens each) + SmolLM2-360M language backbone + multi-modal projector ("connector").

**Training paradigm.** 4-bit NF4 quantization (QLoRA). Adapter is the only trainable component. Constraint: **≤ 5 M trainable parameters**.

**Inference.** Log-likelihood scoring per choice letter — for each candidate `X ∈ {A, B, C, …}`, score `log P(" X" | prompt + "Assistant:")` and take `argmax`. From v8 onwards we also apply Test-Time Augmentation (TTA) by averaging log-probs across original + reversed choice orderings.

**Hardware.** Kaggle T4 (15.6 GB VRAM, fp16 only — bf16 / Flash-Attention 2 unsupported).

**Result summary.**

| # | Test acc. (Kaggle) | Final train loss | Headline change |
|---|---|---|---|
| v2 | 0.61971 | 0.054 | First full run; high LR, narrow LoRA — **overfit** |
| v3 | 0.74446 | 3.373 | LR halved, LoRA targets expanded to MLP |
| v4 | 0.64989 | 1.901 | `TRAIN_WITH_SOLUTION=True` (CoT in user prompt) — **regression** |
| v5 | 0.75653 | 3.338 | CoT reverted, explicit system prompt added |
| v6 | 0.78470 | 2.703 | `IMG_SIZE` 224 → 384, `weight_decay=0.01`, 3 epochs |
| v7 | 0.76659 | 3.604 | Added `out_proj` to LoRA targets — **regression** |
| v8 | 0.80885 / 0.81488 (TTA) | 6.032 | Native chat template (`apply_chat_template`) + `do_image_splitting=True` + BATCH 1×ACCUM 16 |
| v9 | 0.80482 (TTA) | 6.525 | Prompt engineering: doubled question, `Lecture:`/`Hint:` labels, `Answer:` cue — **regression** |
| **v10** | **0.84507 (TTA)** | **5.519** | Image resolution fix (`longest_edge=1024`, no manual thumbnail), simpler prompt restored |

The headline result: **62% → 84.5%** on the public test set across 10 iterations.

---

## 2. v1 / v2 — First end-to-end run

**v1** never produced a submission (smoke run only). **v2** is the first full training.

**Configuration.**
- LoRA: `r = 16`, `alpha = 32`, `dropout = 0.05`, targets `[q_proj, k_proj, v_proj, o_proj]` (attention only, no MLP)
- Training: 3 epochs, BATCH 2 × ACCUM 8 (effective 16), `LR = 2e-4`, `warmup_ratio = 0.1`, cosine schedule, fp16
- Image: `IMG_SIZE = 224`, hard `resize((224, 224))` (no aspect-ratio preservation)
- Prompt: manual ChatML format (`<|im_start|>system / user / assistant <|im_end|>`)
- Collate: only the answer-letter token is labeled; everything else is `-100`
- No `TRAIN_WITH_SOLUTION`, no system prompt yet, no `do_image_splitting`

**Training-loss trajectory.**

| Step | 10 | 20 | 50 | 100 | 200 | 300 | 500 | 580 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 1.344 | 0.881 | 0.467 | 0.335 | 0.268 | 0.221 | 0.135 | **0.054** |

Loss collapsed almost immediately. Test accuracy: **0.61971**.

**Diagnosis.** The model was *memorizing* the answer letter from each (image, choices) pair via the few thousand attention-only LoRA adapters at a high learning rate. There was almost no useful learning signal in the rest of the prompt because (a) only the answer-letter position was labeled and (b) the model never had to generalize beyond the surface pattern of the training samples. The 0.054 final loss with a 62% test score is the textbook signature of pure overfitting.

This run is the most important methodological lesson in the project: **low training loss is not a goal in MCQ fine-tuning**. Every later configuration was designed to keep training loss higher and the model less confident — paradoxically, this is what allowed test accuracy to grow.

---

## 3. v3 — Halve LR, broaden LoRA to MLP layers

**Hypothesis.** Reduce overfitting by giving the LoRA a bigger surface to spread information across, and by slowing optimization.

**Differences vs v2.**
- `LORA_R: 16 → 8`, `LORA_ALPHA: 32 → 16` (same α/r ratio = 2)
- `target_modules`: added `gate_proj`, `up_proj`, `down_proj` (LM MLPs). Now 7 modules instead of 4.
- `LR: 2e-4 → 1e-4`
- `NUM_EPOCHS: 3 → 2`

Same image config (224 hard-resize), same manual ChatML, same letter-only label masking.

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 510 (final) |
|---|---|---|---|---|---|---|
| Loss | 14.420 | 7.630 | 6.253 | 5.553 | 3.862 | **3.373** |

**Test acc: 0.74446** (+12.5 %).

**Interpretation.** Same MCQ task, but the model now has MLP adapters too — much more parameter surface for representing answer patterns rather than memorizing them at the attention layer. Loss is 60× higher than v2 yet test accuracy is sharply better. This confirmed the overfitting diagnosis and set the LoRA-target template (q/k/v/o + gate/up/down) for every subsequent run.

---

## 4. v4 — Chain-of-thought via `TRAIN_WITH_SOLUTION=True`

**Hypothesis.** Including the dataset's `solution` field in the user prompt as `Reasoning: ...` would give the model richer context and improve generalization.

**Differences vs v3.**
- `TRAIN_WITH_SOLUTION = True` — at training time, the user prompt now contains the solution; at inference time, the test set has no solutions.

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 510 (final) |
|---|---|---|---|---|---|---|
| Loss | 10.410 | 4.805 | 3.497 | 2.389 | 1.055 | **1.901** |

**Test acc: 0.64989** (-9.5 %).

**Interpretation.** Loss dropped much faster than v3 — the solution text is essentially a leaked hint, so the model fits the training set easily. But this created a **train/inference distribution shift**: the model learned to depend on a `Reasoning:` block that is absent at test time. This is the classic CoT-leak failure mode for small models. We saved this as a hard "do-not-retry" lesson and reverted in v5.

---

## 5. v5 — Revert CoT, introduce system prompt

**Differences vs v4.**
- `TRAIN_WITH_SOLUTION = False` (reverted)
- First explicit `SYSTEM_PROMPT` introduced: *"You are an expert science educator. Carefully analyze the image and all provided context, then select the single best answer from the choices."*
- Prompt now has a proper `<|im_start|>system\n…<|im_end|>` block.

Everything else identical to v3.

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 510 (final) |
|---|---|---|---|---|---|---|
| Loss | 15.514 | 8.111 | 7.082 | 5.946 | 3.817 | **3.338** |

**Test acc: 0.75653** (+10.7 % over v4, +1.2 % over v3).

**Interpretation.** Recovered v3's level and added a small +1.2 %. The system prompt frames the task more explicitly — minor but real gain.

---

## 6. v6 — Higher image resolution + regularization + 3 epochs

**Hypothesis.** v3-v5 used 224×224 images. SigLIP's native input on the 2B variant is 384×384, so doubling input area should give the vision encoder more detail to attend to. Add weight decay and an extra epoch for headroom.

**Differences vs v5.**
- `IMG_SIZE: 224 → 384`
- `weight_decay = 0.01` (was unset / default 0)
- `NUM_EPOCHS: 2 → 3`

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 500 | 700 | 740 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 16.016 | 8.212 | 6.900 | 5.837 | 3.187 | 5.137 | 3.045 | **2.703** |

**Test acc: 0.78470** (+2.8 %).

**Interpretation.** Largest single jump so far. The visual perception side started mattering once we gave it more pixels. This was the first hint that image-side knobs were a real lever — a thread we returned to (and finally fully exploited) in v10.

---

## 7. v7 — Add `out_proj` to LoRA targets

**Hypothesis.** Including SigLIP's attention output projection in LoRA targets would give us learnable parameters on the *vision* side, not just the LM side.

**Differences vs v6.**
- `target_modules`: added `out_proj` (now 8 modules, total trainable params ≈ 4.93 M, just under the 5 M cap).
- `LECTURE_MAX_CHARS = 1000` (made explicit; previously default).

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 500 | 700 | 750 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 15.135 | 8.064 | 6.881 | 5.605 | 3.201 | 5.354 | 3.132 | **3.604** |

**Test acc: 0.76659** (-1.8 %).

**Interpretation.** Regression. Critical post-mortem: in Idefics3, the name `out_proj` matches **two** modules — SigLIP's attention output **and the multi-modal projector** (the connector that maps visual tokens into the LM embedding space). The connector's pre-trained vision-language alignment is exactly what we should *not* perturb with random LoRA noise. The attempt to fine-tune the visual side via this name accidentally degraded the connector. Lesson saved: future vision-side experiments must target connector modules **explicitly** by their full path, not by an ambiguous suffix.

---

## 8. v8 — Native chat template + image splitting + BATCH 1 / ACCUM 16

This was a structural rewrite, not a single-knob tweak.

**Differences vs v6 (rolling back v7's out_proj).**
- **Chat template fix.** Replaced the manual `<|im_start|>system/user/assistant <|im_end|>` ChatML with `processor.apply_chat_template(...)`. SmolVLM was instruction-tuned on a *different* format (`<|im_start|>User:<image>…<end_of_utterance>\nAssistant:`) — our previous custom ChatML was foreign to the base model, and LoRA capacity was being spent learning the template instead of the science task.
- **`do_image_splitting = True`.** Each image becomes 4 sub-image patches + 1 global thumbnail (5 forward passes through SigLIP) — finer perception of small text in diagrams.
- **`BATCH_SIZE: 2 → 1`, `GRAD_ACCUM: 8 → 16`** (effective batch 16 unchanged; reduced per-mini-batch memory because of the extra visual tokens).
- **Image preprocessing**: `img.resize((384, 384))` → `img.thumbnail((384, 384))` — preserves aspect ratio.
- **TTA introduced at inference time** (choice-permutation: score original + reversed orderings, average per-choice log-probs, argmax).

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 500 | 700 | 770 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 71.040 | 19.216 | 13.289 | 10.394 | 8.379 | 8.133 | 6.208 | **6.032** |

**Test acc: 0.80885 (no TTA), 0.81488 (with TTA)** — first time crossing 80%.

**Interpretation.** Note the dramatic shift in loss scale: starts at 71 (vs 14-16 in v3-v7) and ends at 6 (vs ~3 in v6). This is *not* a sign that training got worse — it's because the new prompt format puts the answer-letter token at a position where the *base* model has very low natural probability (the model wants to start a free-form answer after `Assistant:`, not commit to a single letter). The LoRA had to push much harder against this prior. The fact that loss is much higher and test accuracy is *also* much higher decouples loss-as-a-target from accuracy-as-a-goal once and for all.

A separate val-set diagnostic with this same adapter measured **TTA delivering +2.29 % on val** (81.58 % → 83.87 %), with 54 wrong-→-right corrections vs only 30 right-→-wrong. TTA was confirmed to be a real, reproducible inference-time gain and was kept on for all subsequent submissions.

---

## 9. v9 — Prompt-engineering experiment

**Hypothesis.** Stronger structural cues in the prompt should help the model lock onto the answer position.

**Differences vs v8.**
- Stronger system prompt: *"Respond with ONLY the letter of the correct choice (A, B, C, …) — no explanation, no reasoning, no other text."*
- Question stated **twice**: once before the lecture (so the lecture is read with the question in mind) and once again right before the choices.
- `Lecture:` and `Hint:` are now **separately labeled** instead of merged under a single `Context:` block.
- Explicit `Answer:` cue at the end of the user message, immediately before the chat template's `<end_of_utterance>\nAssistant:`.
- `max_grad_norm = 1.0` (explicit gradient clipping; was unset).
- Plus `do_image_splitting=True` and `thumbnail()` from v8.

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 500 | 700 | 770 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 81.534 | 14.237 | 12.322 | 10.607 | 7.568 | 8.605 | 7.194 | **6.525** |

**Test acc: 0.80482 (TTA)** (-1 % vs v8 with TTA).

**Interpretation.** A controlled negative result. The reformulated prompt sits even further from the model's instruction-tuning distribution than v8's plain prompt — the additional structural cues we introduced were not patterns the base model had seen. The LoRA could partially compensate, but the net effect on test accuracy was slightly negative. We reverted the prompt structure for v10.

A side-finding from this run: a separate experiment with `label_smoothing_factor = 0.1` produced training losses near 110. Investigation revealed that HF Trainer routes `LabelSmoother` with `shift_labels=False` for `Idefics3ForConditionalGeneration` because Idefics3 is registered under `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES`, not `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`. The labels and logits ended up mis-aligned by one position. Permanent rule recorded: **never enable `label_smoothing_factor` with Idefics3**.

---

## 10. v10 — Image-resolution fix (the breakthrough)

**Hypothesis.** Inspecting the official `SmolVLM-500M-Instruct` model card revealed that the 500M variant uses **512×512** patches (not 384) with **64** tokens each, and the recommended default is `size = {"longest_edge": 2048}` (i.e., up to 4×4 = 16 patches + 1 global). All earlier versions were forcing images down to 384×384 *before* handing them to the processor — meaning the input was *smaller* than a single patch, and `do_image_splitting=True` was effectively producing only ~2 patches × 64 tokens ≈ 128 visual tokens per image. The model's native architecture supports up to ~1088 visual tokens. The vision input was being throttled to ~12% of capacity.

**Differences vs v9.**
- **`IMG_SIZE: 384 → 1024`** — but the meaning changed completely: in v9 it was the manual thumbnail target; in v10 it is the processor's `longest_edge` cap.
- `processor.image_processor.size = {"longest_edge": IMG_SIZE}` set explicitly.
- The dataset's `__getitem__` no longer calls `img.thumbnail(...)` — images are passed at natural resolution (the official `Smol_VLM_FT.ipynb` pattern).
- Prompt structure **reverted** to the simpler v8 layout (single `Question:`, merged `Context:` block, no `Answer:` cue, original system prompt).
- `max_grad_norm = 1.0` retained.

This gives roughly 5 patches × 64 tokens ≈ 320 visual tokens per image — about 2.5× v8/v9.

**Training loss.**

| Step | 10 | 50 | 100 | 200 | 300 | 500 | 700 | 770 (final) |
|---|---|---|---|---|---|---|---|---|
| Loss | 64.764 | 14.229 | 11.885 | 9.531 | 8.455 | 7.879 | 5.705 | **5.519** |

The trajectory is below v8 and v9 at every comparable step — the model started lower and converged lower, all because of the visual-input upgrade.

**Test acc: 0.84507 (TTA)** — final submission.

**Interpretation.** This single change (image preprocessing — *not* anything in the LoRA, optimizer, or text prompt) produced the largest test-accuracy jump of the project: **+4.0 %** over v8 (with TTA). It also confirmed that the plateau across v6-v9 was *visual perception capacity*, not LoRA capacity — the model had been visually impaired the entire time, processing roughly 12% of the visual tokens it was designed for.

---

## Loss vs accuracy across all versions

| # | Final train loss | Test acc. | Comment |
|---|---|---|---|
| v2 | 0.054 | 0.620 | catastrophic overfit |
| v3 | 3.373 | 0.744 | broader LoRA spreads signal |
| v4 | 1.901 | 0.650 | CoT leak |
| v5 | 3.338 | 0.757 | recovered |
| v6 | 2.703 | 0.785 | image 224 → 384 |
| v7 | 3.604 | 0.767 | out_proj hits connector |
| v8 | 6.032 | 0.815 (TTA) | new chat template |
| v9 | 6.525 | 0.805 (TTA) | prompt engineering regression |
| v10 | 5.519 | **0.845 (TTA)** | image resolution upgrade |

The **negative correlation** between training loss and test accuracy across v2 → v8 → v10 is the most important pattern in the table:

- v2 had the lowest training loss (0.054) and the lowest test accuracy.
- v8 had a 110× higher training loss (6.0) and a 19-point higher test accuracy.
- v10 dropped training loss within the v8-class regime (5.5 vs 6.0) and gained another 3 points of accuracy.

What changed between these regimes was not the optimization quality — it was *whether the loss was being computed against a meaningful task signal* or against a memorizable surface pattern. Once the prompt format matched the model's instruction tuning (v8) and the image input matched the model's expected resolution (v10), training loss and test accuracy started moving in the same direction again — but at a much higher base loss level than v2's pathological overfit.

---

## Key insights gained

1. **Low training loss is a warning sign for MCQ with single-letter labels.** The single labeled position per sample is so narrow that the model can fit it without learning anything generalizable. Any time the loss collapsed below ~3 in a single epoch, test accuracy regressed.

2. **Match the model's instruction-tuning distribution.** The chat-template fix (v8) was worth ~+2 % despite raising training loss by an order of magnitude. The model's pre-trained distribution is itself a strong prior that custom formats discard.

3. **Image preprocessing was the largest single lever.** Going from `IMG_SIZE=224` to native-resolution `longest_edge=1024` while letting the processor do its own resizing accounted for **roughly 8 percentage points of the total improvement** (v3 to v10). Most of the rest came from text-side / prompt-side changes that each added 1-2 points or stayed flat.

4. **Some plausible-looking experiments fail.** `TRAIN_WITH_SOLUTION=True` (CoT leak), `out_proj` LoRA target (accidental connector tuning), v9's prompt engineering (off-distribution structural cues), `label_smoothing_factor` with Idefics3 (label/logit mis-alignment) all regressed or broke training. The cost of running them is justified only because each one produced a permanent rule for future work.

5. **Test-Time Augmentation is essentially free accuracy.** Choice-permutation TTA (~+1 % consistently, +2.29 % measured on val) requires zero retraining and one wrapper around the scoring function. It compounds with whatever the underlying adapter contributes.

6. **Trust the model card.** The single most expensive lesson of the project was that I assumed the 500M variant of SmolVLM shared the 2B variant's 384×384 patch size. It uses 512×512 with 64 tokens per patch and a default `longest_edge` of 2048. Reading and following the architecture-specific documentation, not the family-level documentation, would have saved several iterations.

---

## Final configuration (v10)

```python
MODEL_ID            = "HuggingFaceTB/SmolVLM-500M-Instruct"
IMG_SIZE            = 1024                              # processor longest_edge
processor.image_processor.size = {"longest_edge": 1024}
do_image_splitting  = True

LORA_R              = 8
LORA_ALPHA          = 16
LORA_DROPOUT        = 0.05
target_modules      = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]   # 7 modules, 4.78 M params

NUM_EPOCHS          = 3
BATCH_SIZE          = 1
GRAD_ACCUM          = 16                                # effective batch = 16
LR                  = 1e-4
lr_scheduler_type   = "cosine"
warmup_ratio        = 0.1
weight_decay        = 0.01
max_grad_norm       = 1.0
fp16                = True

TRAIN_WITH_SOLUTION = False
LECTURE_MAX_CHARS   = 1000
SYSTEM_PROMPT       = "You are an expert science educator. Carefully analyze the image "
                      "and all provided context, then select the single best answer from "
                      "the choices."

# Prompt: processor.apply_chat_template(...) (SmolVLM native format)
# Inference: log-likelihood scoring + TTA (choice permutation)
```

**Final test accuracy: 0.84507**.
