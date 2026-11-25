# Datasets

The pre-download manifest is `configs/datasets_manifest.json` (URL, version, languages, license, access restrictions, speakers, transcript format, sampling rate, intended use, commercial suitability, known limitations). It was written before any download.

| Dataset | Status |
|---|---|
| LJSpeech 1.1 | **Downloaded** (developer-approved 2026-11-18); preprocessed |
| Common Voice, MLS, FLEURS, VCTK | Not downloaded. Common Voice needs a manual terms acceptance on the website; MLS is tens of GB per language; FLEURS is reserved for evaluation; VCTK is for the multi-speaker phase |

## LJSpeech pipeline (`scripts/preprocess_data.py ljspeech`)
1. Archive extracted with Python's `tarfile` `filter="data"` (blocks absolute paths, traversal and special files).
2. Each wav: decode (soundfile) → mono → validate (1-15 s, finite, not silent, clipping ≤ 0.1%, silence ≤ 60%, no silence run > 2.5 s) → resample 22.05 → 16 kHz (soxr_hq) → cache as int16 `.npy`.
3. Transcript: metadata column 3 (pre-normalized) → `normalize_text` → character tokens with blank interspersing.
4. Manifest `data/datasets/ljspeech/manifest.jsonl` (id, text, tokens, duration, peak, RMS dB, clip fraction, silence stats, split) + `stats.json` + `rejected.json`.

**Measured result:** 13,100 clips, 23.92 h, 0 rejected (clean studio-style corpus; the thresholds were not tuned to hit this).
Split (deterministic SHA-1 hash of the utterance id): train 12,609 / val 205 / test 286.

## Why this split
LJSpeech is one speaker, so speaker-disjoint evaluation is **not possible** here; it becomes meaningful with VCTK/MLS. Held-out *utterances* (texts never seen in training) measure generalisation to new text for the known speaker. Hash-based assignment needs no RNG state, so the split is reproducible and stable if the corpus is re-processed.

## Not implemented
Phoneme generation (espeak-ng is not installed; the model uses characters). Pitch/energy extraction (needed only for FastSpeech 2, Milestone 3b).

## Multilingual layout (added 2026-11-24)
`data/datasets/multi/<lang>/{wavs16k/*.npy (int16, 16 kHz), manifest.jsonl, align/, fs2/}`; shared `data/datasets/multi/tokenizer.json`. Manifest row: id, language, speaker, text_raw, text_norm, duration_s, quality stats, split (SHA-1 hash of id: 1.5% test, 1% val). Because audio was cached at 16 kHz, mels computed at 24 kHz have no content above ~8 kHz: the multilingual model's output is band-limited.
| Language | Source | Status (measured) |
|---|---|---|
| en | LJSpeech (existing) | `multi/en`: 5,240 of 13,100 utts (every 5th pair, 9.52 h), symlinks to existing align/fs2/wavs16k, speaker `en_lj` |
| ar | ClArTTS (CC-BY-4.0) | stopped by the user at 4,754 utts / 5.59 h / 1 speaker; **excluded from everything else (user decision)**; data kept untouched |
| zh | AISHELL-3 (Apache-2.0) | OpenSLR served ~15 KB/s (19 GB impossible); using the official `AISHELL/AISHELL-3` Hugging Face mirror, which holds a 3.4 GB / 11,750-wav subset of 200 speaker dirs. text_raw = Hanzi, text_norm = toned pinyin from the corpus (`ni3 hao3`); tone digits are kept as tokens. See README for fetch status |
| de fr it es nl | CML-TTS | no data yet (streaming fetch stalled; main session rewriting it) |
| hi bn ta te mr gu kn ml pa or as ur | FLEURS train (CC-BY-4.0), fetched by the main session with `scripts/fetch_fleurs_indic.py` | pseudo-speakers `<lang>_<gender>` |

Text handling: `normalize_text` now covers nl (letters + numerals via num2words) and zh (a-z0-9 pinyin, no numeral expansion); every character kept by `_KEEP` for en/es/fr/it/de/pt/nl/zh is in the tokenizer (checked programmatically). `CharTokenizer.prepare` romanises hi bn ta te mr gu kn ml pa or as ur ar ko ja with uroman; zh uses pypinyin (toned) for Hanzi input and passes ASCII pinyin through. Indic digits are dropped (only hi has a numeral converter). Dropped-character rates on real data are only measured for en (0.00%); the others are pending data.

## Alignment + readiness (`scripts/prepare_multi.sh`, `scripts/align_multilingual.py`, `ml/training/fs2_data.py:language_coverage`)
Re-runnable; skips finished utterances. A language is trained on only when >= 95% of its manifest rows have both `align/` and `fs2/` files.
Measured zh alignment (jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn, Apache-2.0): 5,437/5,539 aligned (98.2%), 102 failed (Hanzi count != pinyin syllable count, e.g. Latin letters/digits in the text), mean log-prob/frame -0.180, p05 -0.346, median token 48 ms, wall 296 s. Boundaries use spike centres (peaky CTC over whole syllables); aligned speech span is 59% of utterance duration vs 77% energy-based non-silence on a 150-utterance check, i.e. leading/trailing silence is probably over-estimated. English path validated against the existing LJSpeech alignment: mean abs duration difference 0.06 frames over 150 utterances.

### Alignment quality (CPU, measured; `scripts/align_quality.py`, VAD = frames above max(-35 dB, noise floor + 10 dB))
| lang | model | n | token-count mismatch | mean / p05 log-prob per frame | aligned start - VAD start | VAD end - aligned end | speech span / VAD span |
|---|---|---|---|---|---|---|---|
| en (LJSpeech, existing) | wav2vec2-base-960h | 200 | 0 | -0.143 / -0.296 | +0.03 s | +0.07 s | 0.98 |
| de | xlsr-53-german | 200 of 3,619 (all 3,619 aligned, 0 failed) | 0 | -0.106 / -0.221 | +0.05 s | +0.24 s | 0.95 |
| zh | xlsr-53-chinese-zh-cn | 200 | 0 (102 of 5,539 failed earlier on Hanzi/syllable counts) | -0.175 / -0.320 | -0.07 s | +0.20 s | 0.92 |
| hi (candidate test, 100 utts, eval only) | vakyansh-wav2vec2-hindi-him-4200 (MIT) | 100 | 0 failed | -0.104 / -0.214 | +0.14 s | +0.16 s | 0.96 |
Reading: the earlier "59% vs 77%" zh comparison mixed different quantities; with span-vs-VAD the zh and de alignments cover 92-95% of the VAD speech span, with the aligned end 0.2-0.24 s before the energy end (last character's CTC spike precedes its release; en 0.07 s). The trailing-silence token therefore absorbs ~0.2 s of the last phone's tail. zh was NOT re-aligned (training already uses it); `align_multilingual.py` now extends the last zh/ar unit toward the energy end (cap 1 median gap) for new runs, but that change is untested on zh.
