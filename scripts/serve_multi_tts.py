#!/usr/bin/env python
"""Local studio for the multilingual FastSpeech2 model: serves ui/studio.html and a small TTS API.
Run: .venv/bin/python scripts/serve_multi_tts.py [--port 8020] [--run runs/fs2_multi_v1] [--step N]
CPU inference (safe next to training). Acoustic model: own weights; vocoder: pretrained Vocos (MIT)."""
import argparse, glob, io, json, re, subprocess, sys, threading, time, urllib.request, urllib.error, urllib.parse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, soundfile as sf, torch
torch.set_num_threads(3)
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn
from ml.models.fastspeech2.model import FastSpeech2, FS2Config
from ml.preprocessing.tokenizer import CharTokenizer

WORKER = "http://127.0.0.1:8021"      # scripts/natural_worker.py (.venv-nat): Kokoro / Piper / MMS
ORDER = ["en", "de", "fr", "it", "es", "nl", "zh", "hi", "bn", "ta", "te", "mr", "ml", "pa", "or", "as", "ur", "gu", "kn"]
NAMES = {"en": "English", "de": "Deutsch", "fr": "Français", "it": "Italiano", "es": "Español", "nl": "Nederlands", "zh": "中文", "hi": "हिन्दी", "bn": "বাংলা", "ta": "தமிழ்",
         "te": "తెలుగు", "mr": "मराठी", "ml": "മലയാളം", "pa": "ਪੰਜਾਬੀ", "or": "ଓଡ଼ିଆ", "as": "অসমীয়া", "ur": "اردو", "gu": "ગુજરાતી", "kn": "ಕನ್ನಡ"}
# Greetings only (not verified by native speakers).
SAMPLE = {"en": "Hello! Welcome to the PolyVoice studio.", "de": "Hallo! Willkommen im PolyVoice-Studio.", "fr": "Bonjour ! Bienvenue dans le studio PolyVoice.",
          "it": "Ciao! Benvenuti nello studio PolyVoice.", "es": "¡Hola! Bienvenidos al estudio PolyVoice.", "nl": "Hallo! Welkom in de PolyVoice-studio.",
          "zh": "你好！欢迎来到 PolyVoice 工作室。", "hi": "नमस्ते! पॉलीवॉइस स्टूडियो में आपका स्वागत है।", "bn": "নমস্কার! পলিভয়েস স্টুডিওতে আপনাকে স্বাগতম।",
          "ta": "வணக்கம்! பாலிவாய்ஸ் ஸ்டுடியோவிற்கு உங்களை வரவேற்கிறோம்.", "te": "నమస్కారం! పాలీవాయిస్ స్టూడియోకి స్వాగతం.",
          "mr": "नमस्कार! पॉलीव्हॉइस स्टुडिओमध्ये आपले स्वागत आहे.", "ml": "നമസ്കാരം! പോളിവോയ്സ് സ്റ്റുഡിയോയിലേക്ക് സ്വാഗതം.",
          "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਪੌਲੀਵਾਇਸ ਸਟੂਡੀਓ ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ।", "or": "ନମସ୍କାର! ପଲିଭଏସ୍ ଷ୍ଟୁଡିଓକୁ ସ୍ୱାଗତ।",
          "as": "নমস্কাৰ! পলিভয়েচ ষ্টুডিঅ'লৈ আপোনাক স্বাগতম।", "ur": "السلام علیکم! پولی وائس اسٹوڈیو میں خوش آمدید۔",
          "gu": "નમસ્તે! પોલીવોઇસ સ્ટુડિયોમાં તમારું સ્વાગત છે.", "kn": "ನಮಸ್ಕಾರ! ಪಾಲಿವಾಯ್ಸ್ ಸ್ಟುಡಿಯೋಗೆ ಸ್ವಾಗತ."}
NAT_NOTE = {"kokoro": "Pretrained 82M-parameter model.", "piper": "Pretrained single-speaker voice.", "mms": "Pretrained; non-commercial licence, for personal and research use only."}
NAT_SHORT = {"kokoro": "Kokoro", "piper": "Piper", "mms": "Meta MMS"}
# Measured at step 82k-144k (docs/eval_fs2_multi_*.json, 10 held-out sentences/language; synth error vs the SAME sentences' real recordings). Honest, not marketing.
QUALITY = {"en": ("good", "Close to real-speech accuracy in our test."), "it": ("good", "Close to real-speech accuracy in our test."), "es": ("good", "Close to real-speech accuracy in our test."),
           "de": ("fair", "Mostly understandable; some words are wrong."), "fr": ("fair", "Understandable but with frequent word errors."), "nl": ("fair", "Understandable but with frequent word errors."),
           "bn": ("experimental", "Partly intelligible; error rate about 3x real speech."), "te": ("experimental", "Partly intelligible; error rate about 2x real speech."),
           "ur": ("experimental", "Partly intelligible; error rate about 2.5x real speech."), "mr": ("experimental", "Partly intelligible; limited training data."),
           "as": ("experimental", "Partly intelligible; limited training data."), "zh": ("experimental", "Weak: about half the syllables are wrong in our test."),
           "or": ("experimental", "Poor intelligibility; very little training data."), "ml": ("experimental", "Poor intelligibility; limited training data."),
           "hi": ("experimental", "Not reliably measured yet; treat as a preview."), "ta": ("experimental", "Not reliably measured yet; treat as a preview."), "pa": ("experimental", "Not reliably measured yet; treat as a preview.")}

class Engine:
    def __init__(self, run, step):
        ck = (Path(run) / "ckpt" / f"step_{step:07d}.pt") if step else Path(sorted(glob.glob(f"{run}/ckpt/step_*.pt"))[-1])
        s = torch.load(ck, map_location="cpu", weights_only=False); self.net = FastSpeech2(FS2Config(**s["model_cfg"])); self.net.load_state_dict(s["net"]); self.net.eval()
        self.langs, self.smap, self.stats, self.step = s["languages"], s["speaker_map"], s["stats"], s["step"]
        self.tok = CharTokenizer.load(ROOT / "data/datasets/multi/tokenizer.json")
        from vocos import Vocos
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval(); self.lock = threading.Lock()
        self.voices = {l: sorted([k for k in self.smap if k.startswith(l + "_")])[:8] for l in self.langs}
    def ours(self, l):
        if not self.voices.get(l): return None
        q = QUALITY.get(l, ("experimental", ""))
        vs = [{"id": v, "label": ("Female" if v.endswith("_female") else "Male" if v.endswith("_male") else f"Voice {i+1}")} for i, v in enumerate(self.voices[l])]
        return {"id": "ours", "short": "Our model", "name": "PolyVoice (our model)", "license": "own weights; vocoder Vocos (MIT)", "commercial": True, "quality": q[0],
                "badge": {"good": "Good", "fair": "Fair", "experimental": "Experimental"}[q[0]], "note": q[1], "voices": vs}
    def languages(self):
        cat = None
        try: cat = json.load(urllib.request.urlopen(WORKER + "/voices", timeout=3))
        except Exception: pass
        out = []
        for l in ORDER:
            engs = []
            if cat:
                order = {"kokoro": 0, "piper": 0, "mms": 1}
                for e in sorted(cat["languages"].get(l, []), key=lambda e: order.get(e["engine"], 2)):
                    m = cat["engines"][e["engine"]]
                    engs.append({"id": e["engine"], "short": NAT_SHORT.get(e["engine"], m["name"]), "name": m["name"], "license": m["license"], "commercial": m["commercial"], "quality": "natural",
                                 "badge": "Natural" if m["commercial"] else "Natural · non-commercial", "note": NAT_NOTE.get(e["engine"], ""), "voices": e["voices"]})
            o = self.ours(l)
            if o: engs.append(o)
            if engs: out.append({"code": l, "name": NAMES.get(l, l), "sample": SAMPLE.get(l, ""), "engines": engs})
        return out
    @torch.no_grad()
    def synth(self, text, lang, voice, speed, expressive=1.0):
        if lang not in self.langs or voice not in self.smap: raise ValueError("unknown language or voice")
        ids = self.tok.encode(text, lang, add_blank=False)
        if not ids: raise ValueError("Nothing to say after text normalisation (unsupported characters?)")
        t = torch.tensor([[0] + ids + [0]]); t0 = time.perf_counter()
        with self.lock:
            o = self.net(t, torch.tensor([t.size(1)]), sid=torch.tensor([self.smap[voice]]), lid=torch.tensor([self.langs.index(lang)]), speed=speed,
                         pitch_scale=expressive, energy_scale=1.0 + 0.5 * (expressive - 1.0))   # model's predicted pitch varies only ~35-70% as much as real speech; scale restores some of it
            mel = (o["mel"].float() * self.stats["mel_std"] + self.stats["mel_mean"]).transpose(1, 2); a = self.vocos.decode(mel)[0].numpy()
        return a, 24000, time.perf_counter() - t0

class Req(BaseModel):
    text: str; language: str; voice: str; engine: str = "ours"; speed: float = 1.0; expressive: float = 1.0; translate: bool = True; source: str = ""

def make_app(engine):
    app = FastAPI(title="PolyVoice Studio")
    @app.get("/")
    def index(): return FileResponse(ROOT / "ui" / "studio.html")
    @app.get("/api/info")
    def info(): return {"step": engine.step, "languages": engine.languages(), "vocoder": "Vocos mel-24khz (pretrained, MIT) for our own model"}
    @app.post("/api/stt")
    async def stt(request: Request, language: str = ""):
        raw = await request.body()
        if len(raw) < 800 or len(raw) > 12_000_000: raise HTTPException(400, "Recording is empty or too large.")
        p = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-t", "30", "-ar", "16000", "-ac", "1", "-f", "wav", "pipe:1"], input=raw, capture_output=True)
        if p.returncode != 0 or len(p.stdout) < 1000: raise HTTPException(400, "Could not decode the recording.")
        req = urllib.request.Request(WORKER + "/transcribe?" + urllib.parse.urlencode({"language": language}), data=p.stdout, headers={"Content-Type": "application/octet-stream"})
        try:
            with urllib.request.urlopen(req, timeout=900) as resp: return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            try: d = json.loads(e.read()).get("detail", "speech recognition error")
            except Exception: d = "speech recognition error"
            raise HTTPException(400, d)
        except Exception: raise HTTPException(503, "The speech recognition engine is not running (start scripts/natural_worker.py).")
    @app.post("/api/tts")
    def tts(r: Req):
        text = re.sub(r"\s+", " ", r.text).strip()
        if not text or len(text) > 600: raise HTTPException(400, "Enter between 1 and 600 characters.")
        spoken, src, did = text, "", False
        if r.translate:      # type in English (or any supported language); speak in the selected language
            req = urllib.request.Request(WORKER + "/translate", data=json.dumps({"text": text, "target": r.language, "source": r.source}).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=1800) as resp: j = json.loads(resp.read()); spoken, src, did = j["text"], j["source"], j["translated"]
            except urllib.error.HTTPError as e:
                try: d = json.loads(e.read()).get("detail", "translation error")
                except Exception: d = "translation error"
                raise HTTPException(400, d)
            except Exception: raise HTTPException(503, "The translation engine is not running (start scripts/natural_worker.py).")
        text = spoken
        extra = {"X-Spoken-Text": urllib.parse.quote(spoken), "X-Source-Lang": src or "same", "X-Translated": "1" if did else "0"}
        if r.engine != "ours":
            req = urllib.request.Request(WORKER + "/synth", data=json.dumps({"engine": r.engine, "language": r.language, "voice": r.voice, "text": text, "speed": r.speed}).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=900) as resp:
                    return Response(resp.read(), media_type="audio/wav", headers={"X-Gen-Seconds": resp.headers.get("X-Gen-Seconds", "0"), "X-Audio-Seconds": resp.headers.get("X-Audio-Seconds", "0"), "Cache-Control": "no-store", **extra})
            except urllib.error.HTTPError as e:
                try: d = json.loads(e.read()).get("detail", "voice engine error")
                except Exception: d = "voice engine error"
                raise HTTPException(400, d)
            except Exception: raise HTTPException(503, "The natural voice engine is not running (start scripts/natural_worker.py).")
        try: a, sr, dt = engine.synth(text, r.language, r.voice, min(max(r.speed, 0.6), 1.6), min(max(r.expressive, 1.0), 2.2))
        except ValueError as e: raise HTTPException(400, str(e))
        buf = io.BytesIO(); sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav", headers={"X-Gen-Seconds": f"{dt:.2f}", "X-Audio-Seconds": f"{len(a)/sr:.2f}", "Cache-Control": "no-store", **extra})
    return app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8020); ap.add_argument("--run", default="runs/fs2_multi_v1"); ap.add_argument("--step", type=int); a = ap.parse_args()
    eng = Engine(a.run, a.step); print(f"loaded step {eng.step}", flush=True)
    uvicorn.run(make_app(eng), host="127.0.0.1", port=a.port, log_level="warning")
