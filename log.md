# Project Documentation — Whisper Fine-Tuning Log

(Mirrored from Notion "Project Documentation" page, last edited 2026-09-03)

## 1. What this project is

Building a pre-trained speech transcription model that can accurately decipher
Japanese speech from provided video/audio source.

**Goal:** Make an accurate transcription model.
**Success looks like:** Successfully transcribe short audio accurately.

## 2. Roadmap

| Stage | Deliverable | Done when | Status |
|---|---|---|---|
| 1. Inference only | Pretrained Whisper runs on one audio file | Transcript comes out; environment proven | ✅ Done |
| 2. Data pipeline | `Data/prepair_data.py` → WAVs + train/dev manifests | Manifests exist, counts and hours print correctly | ✅ Done |
| 3. Dataset class | `dataset.py` — mel + tokens + collate | One batch prints right shapes; tokens decode back to original text | ✅ Done |
| 4. Training loop | `train.py` | 8–16 samples overfit to near-zero loss | ✅ Done — run1 complete (3 epochs, dev CER 63.57 → 44.66) |
| 5. Evaluation | `evaluate.py` — dev loss + CER | Baseline CER recorded before any training | ⬜ |
| 6. Scale up | More data, bigger model, expressive-speech niche | CER beats the pretrained baseline | ⬜ |

## Stage 2 — Data preparation

### What the script does

1. Transform data from ReazonSpeech (Japanese) into source audio the Whisper
   model can learn on.
2. Split into 3 sets: training, dev (cross-evaluation), test.

### Why each decision was made

- **Why 16 kHz mono?** Matches the format Whisper was originally trained on,
  so transfer learning is not fighting a mismatched input format.
- **Why skip clips longer than 30 seconds?** The encoder has a time limit
  (~30s window in practice); staying under it keeps full context in one pass
  and avoids truncation artifacts.
- **Why write a manifest instead of loading the dataset directly in training?**
  Reading from a manifest of paths/text avoids re-running heavier dataset
  loading/gating logic on every run.
- **Why is the shuffle seeded?** So the dev split is stable across runs —
  otherwise dev CER could shift run-to-run purely from a different dev set,
  not model quality.
- **Why hold out a dev set at all?** To check the model generalizes rather
  than just memorizing training data (overfitting).
- **Why WAV files on disk instead of caching mel spectrograms?** Keeps the
  raw source flexible; caching mels adds constraints for little benefit at
  this scale.
- **Why JSONL instead of a single JSON file?** Line-delimited allows
  streaming/appending without loading the whole file into memory, unlike a
  single `json.load()`-able document.

### Result

| Metric | Value |
|---|---|
| Source dataset | `reazon-research/reazonspeech` (config `tiny`) — gated, access approved |
| Clips kept / skipped | 5323 / 0 |
| Train | 4723 clips · 7.65 h |
| Dev | 300 clips · 0.46 h |
| Test | 300 clips · 0.50 h (held out, non-overlapping — verified) |

## Stage 3 — Dataset class

### What the code does

| Method | What it does |
|---|---|
| `__init__` | Loads JSONL manifest into memory (paths + text only, not audio). Builds Whisper tokenizer once with `multilingual=True, language="ja", task="transcribe"`. |
| `__len__` | Returns number of manifest rows. |
| `__getitem__` | One manifest row → `(mel, tokens)`. Audio: load → 16kHz mono → pad/trim to 30s → log-mel `(80, 3000)`. Text: encode transcript → wrap as `sot_sequence` + text + `eot`. |
| `collate_fn` | List of `(mel, tokens)` → one stacked batch. Mels stack directly. Token sequences padded to batch's longest, returns `mels, input_ids, labels`. |

### Why each decision was made

- **Custom `collate_fn`:** pads token sequences to a common length so they
  can stack into a rectangular tensor; `labels` gets `-100` on pad positions
  so padding doesn't affect the loss/gradient.
- **Mels not padded:** already fixed-size from pad/trim in `__getitem__`.
- **`input_ids` padded with `eot`, `labels` padded with `-100`:** `input_ids`
  is context fed to the model (needs a valid token, so `eot` is the natural
  filler since Whisper has no `<pad>`); `labels` is the answer key, and
  `-100` tells cross-entropy to ignore those positions.
- **Teacher-forcing shift NOT done in `collate_fn`:** collate's job is just
  to produce a padded batch; shifting is a training-loop decision on how to
  use that batch, and doing it twice (once in each place) would make the
  model predict ahead of itself.
- **Pad to batch's longest, not a fixed global length:** cheaper — padding
  everything to a fixed max (e.g. 448) wastes compute versus padding to
  what the batch actually needs.
- **`collate_fn` as a method, not standalone function:** needs `self` to
  access the tokenizer (for `eot`).

### Result

| Check | Value |
|---|---|
| Batch shapes (`batch_size=4`) | mels `(4, 80, 3000)` · input_ids `(4, 112)` · labels `(4, 112)` |
| Padding / masking | Short rows padded to batch max; `labels` has `-100` on pad tail, `input_ids` has `eot` — verified |
| Round-trip decode | All rows decode back to original manifest text |
| Longest transcript (tiny config) | ~112 tokens — well under decoder's 448-position limit |
| Verification script | `scratch_check_batch.py` (throwaway) |

## 4. Decision log

| Date | Decision | Reason | Rejected alternative |
|---|---|---|---|
| 2026-08 | Use `openai-whisper`, not HF `transformers` | See raw model + preprocessing; write training loop by hand | HF `transformers` — kept as later comparison |
| 2026-08 | Ungated parquet mirror instead of official ReazonSpeech | Official repo gated and script-based; `datasets` 5.x can't run loading scripts | Accept gate + second venv with `datasets<4.0` — superseded below |
| 2026-08 | Switched to official `reazon-research/reazonspeech` (`tiny`), downgraded `datasets` to 3.6.0 in main venv | Got Hub access approved; needed script-based loading support removed in `datasets` 5.x | Separate `venv-reazon` — rejected to keep setup simple |
| 2026-08 | Added a third `test` split (300 clips), fixed dev/test/train as non-overlapping consecutive slices | Original two-way split reused dev for monitoring and final eval; needed a clean held-out set | — |
| 2026-08 | Pad token sequences with `eot` in `collate_fn`; mark pad positions `-100` in `labels` | Whisper has no `<pad>` token; `eot` is the natural filler, `-100` makes `cross_entropy` ignore those positions | Fixed global max length — rejected, wastes compute |

## Stage 4 — Training loop

### What the code does

| Block | What it does |
|---|---|
| Setup | Pick device (mps / cpu). Load pretrained `whisper.load_model("tiny")`, `.to(device)`, `.train()`. Build dataset + DataLoader with `collate_fn`. `AdamW` optimizer, small LR (`1e-5`). |
| Loss (per batch) | `logits = model(tokens=input_ids[:, :-1], mel=mels)` → `F.cross_entropy(logits.reshape(-1, V), labels[:, 1:].reshape(-1), ignore_index=-100)`. The `[:-1]`/`[1:]` offset is the teacher-forcing shift, done once here. |
| Step | `opt.zero_grad()` → `loss.backward()` → (grad clip) → `opt.step()`. Order matters — skipping `zero_grad` accumulates gradients across steps. |
| Eval (periodic) | `model.eval()` + `torch.no_grad()`, mean loss over `dev_loader`, then back to `model.train()`. |
| Checkpoint | Save `model.state_dict()` when dev loss improves. |

### Why each decision was made

- **Overfit a single batch first:** deliberate smoke test — if a correctly
  wired model can't memorize 8 examples in 300 steps, something upstream is
  broken (shift, `ignore_index`, gradients, malformed mel/tokens). Cheapest
  possible correctness check before a long run.
- **Small learning rate (`1e-5`):** fine-tuning pretrained weights, not
  training from scratch — large steps would overwrite what the model
  already knows.
- **`ignore_index=-100`:** matches the pad marker set in `labels` by
  `collate_fn`, so padded positions don't contribute to the loss.
- **`model.eval()` + `torch.no_grad()` for dev pass:** `eval()` disables
  dropout/changes norm behavior for consistent scoring; `no_grad()` skips
  building the backward graph since no update happens on dev data.
- **Grad clipping:** prevents one abnormally large gradient from wrecking
  pretrained weights in a single step.

### Result — overfit smoke test

| Check | Value |
|---|---|
| Setup | pretrained `tiny`, 1 batch of 8, `AdamW` lr `1e-5`, 300 steps, reusing same batch |
| Loss curve | step 0: 2.01 → step 20: 0.073 → step 40: 0.007 → … → step 280: 0.0005 |
| Verdict | ✅ Loss collapses to ~0 — full pipeline (dataset → collate → shift → forward → loss → backward → optimizer) is correct |

## 5. Experiment log

| Run | Model | Data | LR / steps | Dev loss | Dev CER | Notes |
|---|---|---|---|---|---|---|
| baseline | tiny (pretrained) | dev (300 clips) | — | — | 63.57% | No training. `evaluate.py`, `beam_size=5`, `language="ja"`, `BasicTextNormalizer`, `fp16=False`. Human CER on JA transcription ≈ 3–5% — that's the ceiling; bottleneck is model capability, not overfitting. |
| run1 | tiny (fine-tuned) | dev (300) | AdamW 1e-5 · 3 epochs · ~1773 steps | 1.029 (step 1200) | 44.66% | `ckpt/best.pt`. −19 pts absolute / ~30% relative vs baseline. Dev loss flattened by ~step 1200 then ticked up — 3rd epoch = mild overfit; best ckpt is from step 1200. Remaining errors: repetition/hallucination loops on short clips + acoustically-plausible kanji swaps. |
| run1 + decode sweep | tiny (fine-tuned, run1) | dev (100 subset) | — (inference only) | — | ~45% (all presets) | Iteration 2 Thread A. Swept beam_size, condition_on_previous_text, temperature fallback, compression/logprob/no_speech thresholds. `current`/`mild`/`no_fallback`/`strict` all ≈ 0.450; `greedy` worse (0.480). Tightening thresholds on the full 300 made it WORSE (49.57%) — stricter thresholds fire the sampling fallback more, model samples more garbage. Conclusion: decoding is a dead end for the hallucination; keep `PRESETS["current"]`. |
| run2 (attempt 1) | tiny (fine-tuned) | dev (300), train filtered `min_duration=1.0` | AdamW 1e-5 · 2 epochs | — | — | Iteration 2 Thread B. Filtered ~78 sub-1s training clips (teach model to emit text into silence). Run hung on `loss.backward()` on MPS after ~1.5h — PyTorch MPS backward is intermittently non-deterministic on this hardware. `ckpt/run2_best.pt` saved but `runs/run2.csv` was never written (loss log writes only at the end of the loop) → run is incomplete, checkpoint is partial. Needs rerun with `PYTORCH_ENABLE_MPS_FALLBACK=1` and/or forced CPU fallback if MPS hangs again. |

**Note:** baseline and fine-tuned numbers must come from the same code path
(same normalizer, same decoding settings, same dev split) or the comparison
isn't valid.

## 6. Environment

- Python 3.11 · venv at `venv/`
- `openai-whisper`, `torch`, `datasets==3.6.0`, `soundfile`, `librosa`,
  `jiwer` (CER metric), system `ffmpeg`
- Gotcha: `datasets` ≥ 4.0 removed script-based loading and
  `trust_remote_code` — downgraded to 3.6.0 to load the gated, script-based
  `reazon-research/reazonspeech`
- Gotcha: audio decoding needs `soundfile` + `librosa` installed separately
- Authenticated via `hf auth login` for gated dataset access
- Gotcha: PyTorch MPS `loss.backward()` intermittently hangs (non-deterministic);
  mitigate with `PYTORCH_ENABLE_MPS_FALLBACK=1`, or force `device="cpu"` if it
  recurs
- Gotcha: Python buffers stdout when redirected to a file — run with `python -u`
  / `PYTHONUNBUFFERED=1` or training progress is invisible until exit
- `MallocStackLogging:` lines on stderr are harmless VS Code-terminal noise

## 7. Known evaluation caveat

`evaluate.py` uses `jiwer.cer()` on text passed through Whisper's
`BasicTextNormalizer`, which is built for English/Latin script. It does not
collapse Japanese kanji/kana script variants (e.g. 私 vs わたし) to a common
form, so CER as currently measured is somewhat overstated — some counted
"errors" are script variation, not real transcription mistakes. This bias is
roughly constant across runs, so run-to-run comparisons (baseline vs run1 vs
run2) remain valid; it only matters for an absolute "how good is this really"
number. A future fix: normalize both reference and hypothesis to a common
form (e.g. kanji→kana via `fugashi`/`MeCab`) before scoring.

## 8. Open questions

- [ ] Is ~8h of training data enough to move CER at all, or only enough to
      prove the loop runs?
- [ ] Clear the ReazonSpeech gate and get the real `small` config?
- [ ] Will longer clips (small / full ReazonSpeech) tokenize past the
      decoder's 448-position limit? Need a length guard in `__getitem__` or
      the manifest builder?
- [ ] Finish run2 retry with MPS mitigations, then `evaluate.py ckpt/run2_best.pt`
      and compare CER + worst-clip loop count against run1's 44.66%.
- [ ] Stage 5 (test-set eval + docs), then bigger model (`base`/`small`) —
      expected to be the largest single CER win.
