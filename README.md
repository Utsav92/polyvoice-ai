# PolyVoice

A local multilingual voice studio. Speak or type in one language, hear it in another, with natural open-licensed voices in 19 languages. The repository also contains a research effort to train a multilingual text-to-speech model from scratch.

**Demo video:** [`demo/polyvoice_demo_curated.mp4`](demo/polyvoice_demo_curated.mp4) (1:30). The translations in the video were written by hand, because the current machine translation handles slang badly; the voices are the real studio voices.

## What works today

| Capability | Engine | Licence |
|---|---|---|
| Voices: en es fr hi it zh | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Apache-2.0 |
| Voices: de nl | [Piper voices](https://huggingface.co/rhasspy/piper-voices) run through onnxruntime (de: Thorsten, nl: Pim / Nathalie) | voice data CC0 |
| Voices: bn ta te mr ml pa or as ur gu kn | Meta MMS-TTS | **CC-BY-NC-4.0, non-commercial only** |
| Translation: en <-> de fr es it nl zh | Helsinki-NLP Opus-MT | Apache-2.0 / CC-BY-4.0 |
| Translation: hi bn ta mr ml gu kn pa or ur | M2M100 (418M) | MIT |
| Speech input (microphone) | Whisper large-v3-turbo | MIT |
| "Our model" (experimental) | multilingual FastSpeech2 trained from scratch here, Vocos vocoder | own weights + MIT |

Pipeline: record or type -> transcribe (Whisper) -> translate -> speak, with the selected engine and voice. Source language is auto-detected, or you can pick it.

## Honest status and limits

- **Our own model is not natural-sounding.** It is intelligible for some languages but robotic and noisy, so the Studio defaults to the pretrained natural voices. Measured with Whisper-small on 10 held-out sentences per language (word error rate of our model's speech, next to real recordings of the same sentences): English 0.11 / 0.01, Italian 0.15 / 0.10, Spanish 0.19 / 0.09, German 0.30 / 0.09, French 0.42 / 0.10, Dutch 0.47 / 0.29, Chinese (scored on pinyin) 0.65 / 0.14. The Indian languages are partly or barely intelligible (see `docs/eval_fs2_multi_*.json`). Causes measured: flat predicted pitch (34-72% of real variation), non-English training audio stored at 16 kHz, and very noisy source recordings for several Indian languages (FLEURS).
- **Telugu and Assamese cannot be translated yet.** The only strong open model for them (IndicTrans2) is gated on Hugging Face. Typing directly in those languages with translation off works.
- **Machine translation mangles slang and idiom**, and may translate brand names literally. Check the "Spoken text" box.
- **Voice input is new.** In a smoke test on clean synthetic clips it transcribed English, German, French, Italian, Spanish, Chinese, Hindi and Tamil almost perfectly, but language auto-detect mistook Bengali and Urdu for Hindi (choose the language in "I'm speaking" to fix it). Real microphone speech and the other Indian languages have not been evaluated; Whisper is generally weaker there.
- **The first use of each model is slow** (it is downloaded and loaded once). The Indian-language voices are non-commercial; the commercial-friendly option (IndicF5, MIT) needs a Hugging Face login and is not wired in yet.
- Non-English voices and Piper use espeak-ng (GPL-3.0) for pronunciation; bundling it in a distributed product needs GPL compliance.

## Run it

Requires Python 3.11, [`uv`](https://docs.astral.sh/uv/), `ffmpeg`, and a few GB of disk for models (downloaded on first use).

```bash
uv venv --python 3.11 .venv      && uv pip install --python .venv/bin/python -e .
uv venv --python 3.11 .venv-nat  && uv pip install --python .venv-nat/bin/python kokoro "misaki[zh]" onnxruntime soundfile fastapi uvicorn transformers \
    uroman pypinyin phonemizer espeakng-loader huggingface_hub numpy sentencepiece sacremoses langdetect protobuf \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
scripts/run_studio.sh        # worker on :8021 (voices, translation, speech recognition) + studio on :8020, then opens the browser
```

Voice input needs microphone permission (works on `http://127.0.0.1`). "Our model" additionally needs a trained checkpoint in `runs/fs2_multi_v1/` (not included). If Hugging Face downloads stall, set `HF_HUB_DISABLE_XET=1`.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/natural_worker.py` | voices (Kokoro, Piper, MMS), translation, speech recognition: one small HTTP service |
| `scripts/serve_multi_tts.py`, `ui/studio.html` | the Studio server and its single-page UI |
| `ml/` | tokenizer, FastSpeech2 and VITS models, trainers, preprocessing, evaluation |
| `scripts/` | data fetching, forced alignment, feature extraction, training and evaluation scripts |
| `configs/` | training configs and `datasets_manifest.json` (every dataset, its licence and approval status) |
| `docs/` | datasets, model licences, training notes, evaluation results |
| `demo/` | the demo video and the scripts that build it (`build_demo.py`, `rendertext.swift`) |
| `backend/`, `frontend/` | an earlier API + Next.js prototype |

## Research notes

- A from-scratch VITS did not learn to use its text input (the prior ignored the text), despite several diagnoses and ablations; see `docs/training.md`. FastSpeech2 with forced-alignment durations works, and is what "Our model" uses.
- **No datasets or weights are included.** Datasets used for training (LJSpeech, CML-TTS, AISHELL-3, ClArTTS, FLEURS) are listed with licences in `configs/datasets_manifest.json` and `docs/datasets.md`.
- Model licences are recorded in `docs/model_licenses.md` (read from Hub metadata, not independently audited).

## Licence

No licence has been chosen for this repository yet, so all rights are reserved until one is added. Third-party models and datasets keep their own licences (see above).
