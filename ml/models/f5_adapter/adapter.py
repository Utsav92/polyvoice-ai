"""F5-TTS adapter. Code MIT; released weights CC-BY-NC-4.0 (Emilia-trained) -> research only. English only (verified claim)."""
from __future__ import annotations
import json, subprocess, tempfile, threading
from pathlib import Path
import soundfile as sf
from ml.models.base import TTSBackend, SynthesisRequest, SynthesisResult

ROOT = Path(__file__).resolve().parents[3]
PY = ROOT / ".venv-f5" / "bin" / "python"

class F5Backend(TTSBackend):
    model_id = "f5-tts"
    display_name = "F5-TTS (research)"
    license = "Code MIT; weights CC-BY-NC-4.0"
    commercial_use = False

    def __init__(self):
        self._proc = None; self._lock = threading.Lock(); self.device = "unknown"

    def is_available(self):
        return (PY.exists(), "" if PY.exists() else "isolated venv .venv-f5 not installed")

    def supported_languages(self): return ["en"]
    def supported_features(self):
        return {"speaking_rate": True, "voice_cloning": True, "requires_reference": True, "emotion": False,
                "streaming": False, "multi_voice": False}

    def load_model(self, language=None):
        if self._proc and self._proc.poll() is None: return
        if not PY.exists(): raise RuntimeError(f"F5 venv missing at {PY}")
        self._proc = subprocess.Popen([str(PY), str(Path(__file__).with_name("worker.py"))], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, cwd=ROOT, bufsize=1)
        hello = json.loads(self._proc.stdout.readline() or "{}")
        if not hello.get("ready"): raise RuntimeError("F5 worker failed to start")
        self.device = hello["device"]

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        if not (req.reference_audio and req.reference_text):
            raise ValueError("F5-TTS requires reference_audio and reference_text (transcript of the reference clip)")
        if not req.text.strip(): raise ValueError("Text is empty")
        with self._lock:
            self.load_model()
            with tempfile.TemporaryDirectory() as td:
                out = str(Path(td) / "o.wav")
                self._proc.stdin.write(json.dumps({"text": req.text, "reference_audio": req.reference_audio,
                    "reference_text": req.reference_text, "speed": req.speaking_rate, "out_path": out}) + "\n")
                self._proc.stdin.flush()
                r = json.loads(self._proc.stdout.readline())
                if not r["ok"]: raise RuntimeError(r["error"])
                audio, sr = sf.read(out, dtype="float32")
        return SynthesisResult(audio, sr, self.model_id, {"device": r["device"], "gen_seconds": r["gen_seconds"], "rtf": r["rtf"]})

    def estimate_resource_usage(self, language=None):
        return {"approx_params_millions": 336, "approx_disk_mb": 1400, "note": "isolated .venv-f5 subprocess; needs reference clip + transcript"}

    def release_model(self):
        if self._proc and self._proc.poll() is None:
            self._proc.stdin.close(); self._proc.terminate(); self._proc.wait(timeout=10)
        self._proc = None
