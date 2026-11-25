"""XTTS-v2 adapter. Weights: Coqui Public Model License (non-commercial). Runs as an isolated subprocess."""
from __future__ import annotations
import json, subprocess, sys, tempfile, threading
from pathlib import Path
import numpy as np, soundfile as sf
from ml.models.base import TTSBackend, SynthesisRequest, SynthesisResult
from ml.preprocessing.text_norm import normalize_text

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv-xtts" / "bin" / "python"
# Languages listed in the XTTS-v2 model card that are also in PolyVoice's target set.
XTTS_LANGS = ["en", "es", "fr", "de", "it", "pt", "ar", "ko", "ja", "hi"]

class XTTSBackend(TTSBackend):
    model_id = "xtts-v2"
    display_name = "Coqui XTTS-v2"
    license = "Coqui Public Model License (non-commercial)"
    commercial_use = False
    default_speaker = "Claribel Dervla"

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.speakers: list[str] = []
        self.device = "cpu"

    def is_available(self):
        return (PY.exists(), "" if PY.exists() else "isolated venv .venv-xtts not installed")

    def supported_languages(self): return list(XTTS_LANGS)
    def supported_features(self):
        return {"speaking_rate": True, "voice_cloning": True, "emotion": False, "streaming": False, "multi_voice": True}

    def load_model(self, language=None):
        if self._proc and self._proc.poll() is None: return
        if not PY.exists(): raise RuntimeError(f"XTTS venv missing at {PY}; run scripts/setup_mac.sh")
        self._proc = subprocess.Popen([str(PY), str(Path(__file__).with_name("worker.py"))], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, cwd=ROOT, bufsize=1)
        hello = json.loads(self._proc.stdout.readline() or "{}")
        if not hello.get("ready"):
            raise RuntimeError("XTTS worker failed to start; see its stderr")
        self.speakers, self.device = hello["speakers"], hello["device"]

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        text = normalize_text(req.text, req.language) if req.language in ("en",) else req.text.strip()
        if not text: raise ValueError("Text is empty after normalization")
        with self._lock:
            self.load_model()
            speaker = req.voice_id or self.default_speaker
            if not req.reference_audio and speaker not in self.speakers:
                raise ValueError(f"Unknown built-in speaker '{speaker}'")
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "o.wav")
                self._proc.stdin.write(json.dumps({"text": text, "language": req.language, "speaker": speaker,
                    "reference_audio": req.reference_audio, "speed": req.speaking_rate, "out_path": out}) + "\n")
                self._proc.stdin.flush()
                r = json.loads(self._proc.stdout.readline())
                if not r["ok"]: raise RuntimeError(r["error"])
                audio, sr = sf.read(out, dtype="float32")
        return SynthesisResult(audio, sr, self.model_id, {"device": self.device, "gen_seconds": r["gen_seconds"], "rtf": r["rtf"]})

    def estimate_resource_usage(self, language=None):
        return {"approx_params_millions": 467, "approx_disk_mb": 1870, "note": "runs in isolated .venv-xtts subprocess, CPU by default"}

    def release_model(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close(); self._proc.terminate(); self._proc.wait(timeout=10)
        self._proc = None
