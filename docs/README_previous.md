# PolyVoice AI

Multilingual speech generation research platform. **Status: Milestones 1–2 complete (incl. follow-ups); Milestone 3 (custom model training) in progress; Milestones 4–6 not started.**

Milestone 3 state: custom VITS (`runs/vits_lj_local`, see docs/training.md) is mid-training, past step 12k of 60k, with intelligibility not yet demonstrated (Whisper back-transcription failed at 10k). Custom English FastSpeech2 (`runs/fs2_lj_local`, 40k steps, Whisper-small WER 0.052 on 30 held-out sentences) works.

## Run locally (Apple Silicon, tested on M5 Pro / 24 GB)
    uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e .
    # optional XTTS (needs CPML acceptance, ~1.8 GB):
    uv venv --python 3.11 .venv-xtts && uv pip install --python .venv-xtts/bin/python "coqui-tts[codec,languages]" torch torchaudio soundfile
    .venv/bin/uvicorn backend.app.main:app --port 8010
    cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8010 npm run dev
    .venv/bin/python -m pytest tests -q        # real-model tests; first run downloads ~145 MB

Requires ffmpeg on PATH (MP3 export). Device is auto-detected (CUDA > MPS > CPU); override with POLYVOICE_DEVICE.

## Implemented
- Backends: `mms-tts` (en hi es fr de ar ko pt) and `xtts-v2` (all 10 target languages, isolated `.venv-xtts` subprocess, CPU). Both **non-commercial**; see docs/model_licenses.md
- `f5-tts` (English only, research; needs reference clip + transcript, so disabled in the UI until voice enrollment exists; own `.venv-f5`)
- 58 XTTS built-in voices with real sample clips (`scripts/build_voice_manifest.py`), `/api/v1/voices`
- Numeral expansion for en/es/fr/de/pt/it/ar/ko/ja (num2words) and hi (own converter)
- `/api/v1/languages` capability matrix derived from adapters; language-first UI
- Capability registry rejecting unsupported language/emotion/rate combinations
- Async job API (`/tts/generate`, `/jobs`, `/jobs/{id}`, `/audio/{id}`, `/models`, `/health`), SQLite storage, WAV/MP3
- Speech Studio UI (model/language/speed/format, waveform, player, download)

## Not implemented yet
Auth, rate limiting, PostgreSQL (SQLite for now; no psql/Docker installed on this machine), voice cloning, translation, audiobooks, custom VITS/FastSpeech2 training, other languages, evaluation, Docker, remaining frontend pages.

## Known limitations
- Korean numerals use Sino-Korean readings ("삼 개"); natural counting would use native Korean ("세 개").
- Japanese XTTS back-transcription CER 0.278 on one sentence (greeting garbled); Arabic MMS 0.227. Single-sentence checks, not benchmarks.
- XTTS runs on CPU (MPS unreliable); F5 runs on MPS at RTF ~1.1-1.5 (slower than real time).
- Frontend uses webpack (`--webpack`) with `@tailwindcss/postcss`: Turbopack's production build served a stale stylesheet and ignored CSS edits in this environment.
- torchaudio>=2.9 needs shared FFmpeg libs via torchcodec; workers patch `torchaudio.load` to use soundfile.

## Multilingual FastSpeech2 (opt-in, in progress)
- Config `configs/fs2_multi.yaml`: fixed language list (en de fr it es nl zh hi bn ta te mr gu kn ml pa or as ur; no ar/ja/sv), 1024 fixed speaker slots (`<run_dir>/speaker_map.json`, append-only), language-balanced sampling (share ~ hours^0.5), normalisation stats frozen at first start in `<run_dir>/fs2_stats_frozen.json`. Languages without alignment+features (>=95% of manifest) are skipped and picked up on restart. Resume with added languages verified in a short test (runs/fs2_multi_resume_test).
- Scripts: `scripts/align_multilingual.py` (Apache-2.0 per-language CTC aligners), `scripts/prepare_multi.sh` (align + features, re-runnable), `scripts/fetch_aishell3.py`. Verified: zh alignment, English alignment path, 7 short smoke/resume runs. Not done: aligners for de/fr/it/es/nl beyond zh at time of writing (hub transfers slow), no permissive aligner chosen for Indic languages, no listening/WER evaluation of any multilingual output.
- Caveat: non-English audio is cached at 16 kHz, so synthesized audio is band-limited (~8 kHz).
