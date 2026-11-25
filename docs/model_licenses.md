# Model licenses (verified 2026-11-18)

| Backend | Weights license | Commercial use | Notes |
|---|---|---|---|
| Meta MMS-TTS (`facebook/mms-tts-*`) | CC-BY-NC-4.0 (read from Hub card metadata) | **No** | One checkpoint per language. No checkpoint exists for `jpn` or `ita` (Hub returned RepositoryNotFound). |
| Coqui XTTS-v2 | Coqui Public Model License (CPML) | **No** | The developer explicitly accepted the CPML for non-commercial research on 2026-11-18 before the ~1.8 GB download. `COQUI_TOS_AGREED=1` is set only in `ml/models/xtts_adapter/worker.py` for that reason. Do not deploy commercially. |
| AI4Bharat IndicF5 (`ai4bharat/IndicF5`) | MIT (read from Hub card metadata 2026-11-19) | Yes (model licence; its training data is indicvoices_r and Rasa, both cc-by-4.0 per HF metadata) | Gated (auto-approval; needs HF login + accepted terms). 11 languages: as bn gu hi kn ml mr or pa ta te (no Urdu). Developer approved 2026-11-19 to generate SYNTHETIC training audio with it for the Indian-language VITS. Needs reference clip + transcript per voice. Not yet downloaded. |

Whisper-small (`openai/whisper-small`, Apache-2.0) is used only for evaluation back-transcription.

XTTS built-in speakers (58 names, e.g. "Claribel Dervla") ship with the model under the CPML. Their provenance is Coqui's; PolyVoice makes no claim about them beyond that.

## Multilingual FastSpeech 2 pipeline (added 2026-11-24)
| Component | Licence | Use | Notes |
|---|---|---|---|
| Vocos `charactr/vocos-mel-24khz` | MIT | vocoder at inference (pretrained, not ours) | unchanged |
| `facebook/wav2vec2-base-960h` | Apache-2.0 | English duration labels | unchanged |
| `jonatasgrosman/wav2vec2-large-xlsr-53-{german,french,italian,spanish,dutch,chinese-zh-cn}` | Apache-2.0 (Hub metadata, checked 2026-11-24) | CTC forced alignment = duration LABELS only (de fr it es nl zh); fine-tuned from XLSR-53 on Common Voice (CC0) | chosen because Meta MMS / torchaudio `MMS_FA` are CC-BY-NC-4.0 and were NOT used. Arabic model is out of scope (Arabic dropped by the user). **Not yet downloaded/run** (Hub transfer stalled, see README) |
| Indic CTC aligners | not chosen | - | candidates with permissive Hub licence tags: `ai4bharat/indicwav2vec-hindi` (apache-2.0), `ai4bharat/indicwav2vec_v1_bengali` (mit), `Harveenchadha/vakyansh-wav2vec2-{hindi-him-4200,tamil-tam-250}` (mit). Licence tags are not the same as training-data licence; unverified, no coverage for te mr gu kn ml pa or as ur found. Needs a decision |
| `uroman` (pip) | Apache-2.0 (package classifier) | romanisation of hi bn ta te mr gu kn ml pa or as ur ar zh(fallback) text | |
| `pypinyin` | MIT | Hanzi -> toned pinyin for zh inference text | training pinyin comes from the AISHELL-3 transcripts |
| `num2words` | LGPL (package classifier) | numeral expansion (de fr it es nl en ar ko ja pt) | used as a library |

### Indic CTC aligner candidates (Hub licence tags read via the API on 2026-11-24; labels only, not shipped)
Permissive tag present: hi `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (mit, 378 MB), `infinitejoy/wav2vec2-large-xls-r-300m-hindi` (apache-2.0); `ai4bharat/indicwav2vec-hindi` (apache-2.0 but files return 401, not downloadable unauthenticated); bn `arijitx/wav2vec2-xls-r-300m-bengali`, `shahruk10/wav2vec2-xls-r-300m-bengali-commonvoice` (apache-2.0); ta `Harveenchadha/vakyansh-wav2vec2-tamil-tam-250` (mit), `Amrrs/wav2vec2-large-xlsr-53-tamil`, `Thanish/wav2vec2-large-xlsr-tamil` (apache-2.0); te `anuragshas/wav2vec2-large-xlsr-53-telugu`, `krishnateja/wav2vec2-telugu` (apache-2.0); mr `sumedh/wav2vec2-large-xlsr-marathi`, `infinitejoy/wav2vec2-large-xls-r-300m-marathi-cv8` (apache-2.0); ml `gvs/wav2vec2-large-xlsr-malayalam`, `kurianbenoy/wav2vec2-large-xls-r-300m-malayalamv1` (apache-2.0); pa `Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10` (mit), `kingabzpro/wav2vec2-large-xlsr-53-punjabi` (apache-2.0); or `theainerd/...odia`, `anuragshas/...odia`, `Harveenchadha/odia_large_wav2vec2` (apache-2.0); as `jayashreedevi2020/wav2vec2-large-xls-r-300m-assamese`, `manandey/wav2vec2-large-xlsr-assamese` (apache-2.0); ur `kingabzpro/wav2vec2-large-xls-r-300m-Urdu`, `kingabzpro/wav2vec2-urdu` (apache-2.0).
**No licence-verified candidate: Kannada and Gujarati.** Only `Harveenchadha/vakyansh-wav2vec2-{kannada-knm-560,gujarati-gnm-100}` (no model card, no licence tag; the sibling Vakyansh Hindi/Tamil/Punjabi models are tagged MIT), `anvitamanne/wav2vec2-xlsr-53-kannada` (card says "More Information Needed") and `taneeshk12/Gujarati_wav2vec2_kathbath` (no card); `jaimin/wav2vec2-base-gujarati-demo` is apache-2.0 tagged but a demo. Not used and not evaluated; needs a user decision. A tag is not a verification of the fine-tuning data's licence.
Selection among candidates is made by measured alignment quality (`scripts/align_multilingual.py <lang> --model <id> --eval 100`); the results table is in docs/datasets.md.

## Natural-voice engines used by PolyVoice Studio (added 2026-11-25; licences read from Hub metadata / model cards, not independently audited)
| Engine | Languages | Licence | Commercial | Notes |
|---|---|---|---|---|
| Kokoro-82M (`hexgrad/Kokoro-82M`) | en es fr hi it zh | Apache-2.0 (Hub tag) | Yes (model) | Ungated. Non-English text is phonemised with espeak-ng (GPL-3.0): bundling it in a distributed product needs GPL compliance. |
| Piper voices via onnxruntime (`rhasspy/piper-voices`) | de: Thorsten high (dataset CC0); nl: Pim medium (CC0), Nathalie nl_BE medium (CC0) | Voice data per model card (CC0 here); repo tag MIT | Yes (data) | Own minimal onnxruntime runtime in `scripts/natural_worker.py` (no GPL Piper package); phonemisation again via espeak-ng. |
| Meta MMS-TTS (`facebook/mms-tts-*`) | bn ta te mr ml pa or as ur gu kn (hi also) | CC-BY-NC-4.0 | **No** | Stopgap for Indian languages; the Studio labels it "Natural · non-commercial". Replace with IndicF5 (MIT, gated; needs HF login + reference clips) for commercial use. |
Our own multilingual FastSpeech2 remains available as "Our model" in the Studio (experimental; see docs/eval_fs2_multi_*.json).
