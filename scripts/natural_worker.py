#!/usr/bin/env python
"""Natural-voice worker: pretrained, permissively licensed TTS engines behind one tiny HTTP API (run with .venv-nat).
  Kokoro-82M (Apache-2.0)            en es fr hi it zh
  Piper voices run via onnxruntime   de (Thorsten, CC0 data), nl (Pim, CC0 data)
  Meta MMS-TTS (CC-BY-NC-4.0, NON-COMMERCIAL, stopgap)   bn ta te mr ml pa or as ur gu kn
Run: HF_HUB_DISABLE_XET=1 .venv-nat/bin/python scripts/natural_worker.py [--port 8021]
GET /voices -> engines + voices;  POST /synth {engine, language, voice, text, speed} -> audio/wav."""
import argparse, io, json, re, threading, time
from pathlib import Path
import numpy as np, soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn

KOKORO_LANG = {"en": "a", "es": "e", "fr": "f", "hi": "h", "it": "i", "zh": "z"}
KOKORO_VOICES = {"en": ["af_heart", "af_bella", "af_nicole", "am_michael", "am_fenrir", "bf_emma", "bm_george"], "es": ["ef_dora", "em_alex"], "fr": ["ff_siwis"],
                 "hi": ["hf_alpha", "hf_beta", "hm_omega", "hm_psi"], "it": ["if_sara", "im_nicola"], "zh": ["zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zm_yunxi", "zm_yunjian"]}
PIPER = {"de": [("Thorsten (high)", "de/de_DE/thorsten/high/de_DE-thorsten-high")], "nl": [("Pim (medium)", "nl/nl_NL/pim/medium/nl_NL-pim-medium"), ("Nathalie, Flemish (medium)", "nl/nl_BE/nathalie/medium/nl_BE-nathalie-medium")]}
MMS = {"bn": "ben", "ta": "tam", "te": "tel", "mr": "mar", "ml": "mal", "pa": "pan", "or": "ory", "as": "asm", "ur": "urd-script_arabic", "gu": "guj", "kn": "kan", "hi": "hin"}

class Kokoro:
    def __init__(self): self.p = {}; self.lock = threading.Lock()
    def synth(self, lang, voice, text, speed):
        from kokoro import KPipeline
        with self.lock:
            if lang not in self.p: self.p[lang] = KPipeline(lang_code=KOKORO_LANG[lang], repo_id="hexgrad/Kokoro-82M")
            chunks = [np.asarray(a, dtype="float32") for _, _, a in self.p[lang](text, voice=voice, speed=speed) if a is not None]
        if not chunks: raise ValueError("nothing to synthesise")
        return np.concatenate(chunks), 24000

class Piper:
    def __init__(self): self.m = {}; self.lock = threading.Lock()
    def _load(self, path):
        if path not in self.m:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            onnx = hf_hub_download("rhasspy/piper-voices", path + ".onnx"); cfg = json.load(open(hf_hub_download("rhasspy/piper-voices", path + ".onnx.json")))
            self.m[path] = (ort.InferenceSession(onnx, providers=["CPUExecutionProvider"]), cfg)
        return self.m[path]
    def synth(self, lang, voice, text, speed):
        import espeakng_loader
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        from phonemizer import phonemize
        EspeakWrapper.set_library(espeakng_loader.get_library_path()); EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
        sess, cfg = self._load(voice)
        ph = phonemize(text, language=cfg["espeak"]["voice"], backend="espeak", strip=True, preserve_punctuation=True, with_stress=True)
        pm = cfg["phoneme_id_map"]; ids = list(pm["^"])
        for ch in ph:
            if ch in pm: ids += pm[ch] + pm["_"]
        ids += pm["$"]; inf = cfg.get("inference", {}); scales = np.array([inf.get("noise_scale", .667), inf.get("length_scale", 1.0) / speed, inf.get("noise_w", .8)], dtype=np.float32)
        feeds = {"input": np.array([ids], dtype=np.int64), "input_lengths": np.array([len(ids)], dtype=np.int64), "scales": scales}
        if "sid" in [i.name for i in sess.get_inputs()]: feeds["sid"] = np.array([0], dtype=np.int64)
        with self.lock: audio = sess.run(None, feeds)[0].squeeze().astype("float32")
        return audio, cfg["audio"]["sample_rate"]

class Mms:
    def __init__(self): self.m = {}; self.lock = threading.Lock()
    def synth(self, lang, voice, text, speed):
        import torch
        from transformers import VitsModel, AutoTokenizer
        code = MMS[lang]
        if code not in self.m: self.m[code] = (VitsModel.from_pretrained(f"facebook/mms-tts-{code}").eval(), AutoTokenizer.from_pretrained(f"facebook/mms-tts-{code}"))
        model, tok = self.m[code]
        if getattr(tok, "is_uroman", False):
            import uroman; text = uroman.Uroman().romanize_string(text)
        inp = tok(text, return_tensors="pt")
        with self.lock, torch.no_grad():
            model.speaking_rate = speed; w = model(**inp).waveform[0].numpy().astype("float32")
        return w, model.config.sampling_rate

ENG = {"kokoro": Kokoro(), "piper": Piper(), "mms": Mms()}
META = {"kokoro": ("Kokoro", "Apache-2.0", True), "piper": ("Piper", "CC0 / CC-BY voice data, MIT runtime", True), "mms": ("Meta MMS-TTS", "CC-BY-NC-4.0 (non-commercial)", False)}

def catalog():
    langs = {}
    for l, vs in KOKORO_VOICES.items(): langs.setdefault(l, []).append({"engine": "kokoro", "voices": [{"id": v, "label": v.split("_")[1].title() + (" ♀" if v[1] == "f" else " ♂")} for v in vs]})
    for l, vs in PIPER.items(): langs.setdefault(l, []).append({"engine": "piper", "voices": [{"id": p, "label": n} for n, p in vs]})
    for l in MMS: langs.setdefault(l, []).append({"engine": "mms", "voices": [{"id": "default", "label": "Default"}]})
    return {"engines": {k: {"name": n, "license": lic, "commercial": c} for k, (n, lic, c) in META.items()}, "languages": langs}


# ---------------------------------------------------------------- translation (open licences: Opus-MT Apache-2.0/CC-BY-4.0, M2M100 MIT)
OPUS_FROM_EN = {"de": "Helsinki-NLP/opus-mt-en-de", "fr": "Helsinki-NLP/opus-mt-en-fr", "es": "Helsinki-NLP/opus-mt-en-es", "it": "Helsinki-NLP/opus-mt-en-it", "nl": "Helsinki-NLP/opus-mt-en-nl", "zh": "Helsinki-NLP/opus-mt-en-zh"}
OPUS_TO_EN = {"de": "Helsinki-NLP/opus-mt-de-en", "fr": "Helsinki-NLP/opus-mt-fr-en", "es": "Helsinki-NLP/opus-mt-es-en", "it": "Helsinki-NLP/opus-mt-it-en", "nl": "Helsinki-NLP/opus-mt-nl-en", "zh": "Helsinki-NLP/opus-mt-zh-en"}
M2M_LANGS = {"hi", "bn", "ta", "mr", "ml", "gu", "kn", "pa", "or", "ur"}      # M2M100 has no Telugu or Assamese (IndicTrans2 is gated: needs an HF login)
SCRIPT = {"zh": (0x4E00, 0x9FFF), "hi": (0x0900, 0x097F), "mr": (0x0900, 0x097F), "bn": (0x0980, 0x09FF), "as": (0x0980, 0x09FF), "pa": (0x0A00, 0x0A7F), "gu": (0x0A80, 0x0AFF),
          "or": (0x0B00, 0x0B7F), "ta": (0x0B80, 0x0BFF), "te": (0x0C00, 0x0C7F), "kn": (0x0C80, 0x0CFF), "ml": (0x0D00, 0x0D7F), "ur": (0x0600, 0x06FF)}
NAMES_EN = {"en": "English", "de": "German", "fr": "French", "it": "Italian", "es": "Spanish", "nl": "Dutch", "zh": "Chinese", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
            "mr": "Marathi", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ur": "Urdu", "gu": "Gujarati", "kn": "Kannada"}
_TM = {}; _TLOCK = threading.Lock()

def detect_lang(text):
    letters = [c for c in text if c.isalpha()]
    if not letters: return "en"
    for code, (lo, hi) in SCRIPT.items():
        if sum(lo <= ord(c) <= hi for c in letters) / len(letters) > 0.5:
            return "hi" if code in ("hi", "mr") and False else code      # Devanagari/Bengali scripts are shared by several languages: keep the script's first language
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
    try: d = detect(text)
    except Exception: return "en"
    return {"zh-cn": "zh", "zh-tw": "zh"}.get(d, d)

def _split(text):
    parts = [x.strip() for x in re.split(r"(?<=[.!?。！？])\s+", text) if x.strip()]
    return parts or [text]

def _opus(name, sents):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    if name not in _TM: _TM[name] = (AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name).eval())
    tok, m = _TM[name]
    with torch.no_grad(): out = m.generate(**tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256), num_beams=4, max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)

def _m2m(sents, src, tgt):
    import torch
    from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
    if "m2m" not in _TM: _TM["m2m"] = (M2M100Tokenizer.from_pretrained("facebook/m2m100_418M"), M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").eval())
    tok, m = _TM["m2m"]; tok.src_lang = src
    with torch.no_grad(): out = m.generate(**tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=256), forced_bos_token_id=tok.get_lang_id(tgt), num_beams=4, max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)

def translate(text, target, source=None):
    src = source or detect_lang(text)
    if src == target: return text, src, False
    sents = _split(text)
    with _TLOCK:
        if target == "en" or src == "en":
            if src == "en":
                if target in OPUS_FROM_EN: out = _opus(OPUS_FROM_EN[target], sents)
                elif target in M2M_LANGS: out = _m2m(sents, "en", target)
                else: raise ValueError(f"Translation into {NAMES_EN.get(target, target)} is not available yet (needs IndicTrans2, which requires a Hugging Face login). Turn translation off to read text as typed.")
            else:
                if src in OPUS_TO_EN: out = _opus(OPUS_TO_EN[src], sents)
                elif src in M2M_LANGS: out = _m2m(sents, src, "en")
                else: raise ValueError(f"Cannot read the input language ({src}). Type in English or in the selected language.")
        else:   # pivot through English
            en = _opus(OPUS_TO_EN[src], sents) if src in OPUS_TO_EN else _m2m(sents, src, "en") if src in M2M_LANGS else None
            if en is None: raise ValueError(f"Cannot read the input language ({src}). Type in English or in the selected language.")
            if target in OPUS_FROM_EN: out = _opus(OPUS_FROM_EN[target], en)
            elif target in M2M_LANGS: out = _m2m(en, "en", target)
            else: raise ValueError(f"Translation into {NAMES_EN.get(target, target)} is not available yet (needs IndicTrans2, which requires a Hugging Face login).")
    return ("" if target == "zh" else " ").join(out).strip(), src, True

class TReq(BaseModel):
    text: str; target: str; source: str = ""


# ---------------------------------------------------------------- speech recognition (Whisper large-v3-turbo, MIT)
ASR_LANGS = {"en", "de", "fr", "it", "es", "nl", "zh", "hi", "bn", "ta", "te", "mr", "ml", "pa", "gu", "kn", "ur", "as"}      # Whisper has no Odia
_ASR = {}; _ALOCK = threading.Lock()
def transcribe(audio, language=None):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    name = "openai/whisper-large-v3-turbo"
    if "m" not in _ASR: _ASR["m"] = (WhisperProcessor.from_pretrained(name), WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=torch.float32).eval())
    proc, model = _ASR["m"]
    feats = proc(audio, sampling_rate=16000, return_tensors="pt").input_features
    with _ALOCK, torch.no_grad():
        if not language:
            tid = model.detect_language(input_features=feats)[0].item(); language = proc.tokenizer.decode([tid]).strip("<|>")
        ids = model.generate(input_features=feats, language=language, task="transcribe", num_beams=4, max_new_tokens=220)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip(), language

class Req(BaseModel):
    engine: str; language: str; voice: str; text: str; speed: float = 1.0

app = FastAPI(title="PolyVoice natural worker")
@app.get("/voices")
def voices(): return catalog()
@app.post("/transcribe")
async def transcribe_ep(request: Request, language: str = ""):
    data = await request.body(); t0 = time.perf_counter()
    try: y, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception: raise HTTPException(400, "Could not read the audio (expected 16 kHz mono WAV).")
    if y.ndim > 1: y = y.mean(1)
    if sr != 16000: raise HTTPException(400, "Audio must be 16 kHz.")
    if len(y) < 4000: raise HTTPException(400, "That recording is too short.")
    if float(np.abs(y).max()) < 0.003: raise HTTPException(400, "No sound was picked up. Check your microphone.")
    y = y[:16000 * 30]
    try: text, lang = transcribe(y, language or None)
    except Exception as e: raise HTTPException(400, f"Could not transcribe: {type(e).__name__}: {str(e)[:120]}")
    return {"text": text, "language": lang, "supported": lang in ASR_LANGS, "seconds": round(time.perf_counter() - t0, 2), "audio_seconds": round(len(y) / 16000, 1)}
@app.post("/translate")
def translate_ep(r: TReq):
    t0 = time.perf_counter()
    try: out, src, did = translate(r.text.strip(), r.target, r.source or None)
    except ValueError as e: raise HTTPException(400, str(e))
    return {"text": out, "source": src, "translated": did, "seconds": round(time.perf_counter() - t0, 2)}
@app.post("/synth")
def synth(r: Req):
    if r.engine not in ENG: raise HTTPException(400, "unknown engine")
    t0 = time.perf_counter()
    try: a, sr = ENG[r.engine].synth(r.language, r.voice, re.sub(r"\s+", " ", r.text).strip(), min(max(r.speed, .6), 1.6))
    except (ValueError, KeyError) as e: raise HTTPException(400, f"{type(e).__name__}: {e}")
    buf = io.BytesIO(); sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav", headers={"X-Gen-Seconds": f"{time.perf_counter()-t0:.2f}", "X-Audio-Seconds": f"{len(a)/sr:.2f}", "Cache-Control": "no-store"})

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8021); a = ap.parse_args(); uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
