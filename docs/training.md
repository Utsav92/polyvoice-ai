# Training the custom VITS

Architecture (`ml/models/vits/`), all trained **from random initialization** (no pretrained components):
text encoder (4-layer transformer, sinusoidal positions, conv FFN) → monotonic alignment search (numba) + deterministic log-duration predictor → posterior encoder (WaveNet, 8 layers) → 4-layer residual-coupling normalizing flow → HiFi-GAN decoder (init channel 256) → multi-period + multi-scale discriminators (5 periods + 1 scale, GAN + feature-matching + mel L1 + KL losses). Optional speaker/language embeddings are implemented but unused in the single-speaker run. **10.83 M parameters** (decoder 3.76 M).

Deviations from the original VITS, chosen for stability/cost on a Mac mini: character input (no phonemizer), deterministic instead of stochastic duration predictor, standard attention instead of relative-position attention, 16 kHz audio, fewer/narrower layers, and a GAN warm-up (adversarial terms off until step 20,000).

## Commands
    .venv/bin/python scripts/preprocess_data.py ljspeech
    .venv/bin/python scripts/train_vits.py --config configs/local_mps.yaml      # resumes from the latest checkpoint
    .venv/bin/python scripts/evaluate.py --models mms-tts polyvoice-vits-lj --n 30
Configs: `local_mps.yaml` (tested), `cloud_cuda.yaml` and `cloud_large.yaml` (**untested**; fp32 only, mixed precision not implemented).

## Measured on Apple M5 Pro (24 GB), batch 16, 60-step benchmark (`docs/training_benchmark_mps.jsonl`)
- Without adversarial terms: 0.23-0.33 s/step, ~330 MB MPS memory.
- With adversarial terms: 0.80-0.96 s/step, ~830 MB.
- A full 60,000-step run therefore needs on the order of 11 h of wall time.

## Status (updated as the run progresses)
Run `runs/vits_lj_local`, `configs/local_mps.yaml`, started under `caffeinate`, target 60,000 steps (adversarial terms from step 20,000).

| Step | val mel L1 | Whisper-small back-transcription of the 2 fixed sample sentences |
|---|---|---|
| 2,000 | 0.641 | not intelligible (hallucinated unrelated sentences, WER > 1) |
| 10,000 | 0.523 | **not intelligible** (unrelated text; one sample degenerates into a repetition loop) |

Training diagnostics, mean per 1k-step window (steps 0-1k / 2-3k / 5-6k / 9-10k): mel 0.84 / 0.64 / 0.58 / 0.54; alignment log-likelihood -311 / -294 / -280 / -272 (improving); duration error ~1.15 frames per token (flat); output length tracks text length (2.8 s and 7.0 s for 44- and 109-character inputs).
Interpretation: the alignment, duration and acoustic paths are learning, but **intelligibility has not been demonstrated**. 10k steps is far below what full-size VITS needs, and this model has not yet entered its adversarial phase. Do not treat the custom model as working until `scripts/evaluate.py` shows a low WER at a later checkpoint.

## Known limitations
Gradient accumulation sums gradients over micro-batches and then steps both optimizers once, so the generator sees a discriminator that is one accumulation window stale (identical to standard VITS when `grad_accum: 1`). Training log fields: loss terms, `dur_mae_frames` (duration predictor error vs MAS) and `mas_ll` (alignment log-likelihood) as alignment diagnostics, MPS memory, step time, epoch time, checkpoint size.
