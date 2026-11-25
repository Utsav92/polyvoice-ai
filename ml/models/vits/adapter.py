"""Backend for the custom scaled-down VITS trained from random initialization in this repo (see docs/training.md)."""
from __future__ import annotations
import glob, os, time
from pathlib import Path
import numpy as np, torch
from ml.models.base import TTSBackend, SynthesisRequest, SynthesisResult
from ml.inference.device import detect_device
from ml.models.vits.model import VITS, VitsConfig
from ml.preprocessing.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[3]

def _tupleize(d):
    return {k: (tuple(tuple(i) if isinstance(i, list) else i for i in v) if isinstance(v, list) else v) for k, v in d.items()}

class VitsBackend(TTSBackend):
    model_id = "polyvoice-vits-lj"
    display_name = "PolyVoice VITS (custom, LJSpeech)"
    license = "Own weights, random-init; trained on LJSpeech (public domain per dataset page)"
    commercial_use = True   # no third-party weights; re-verify dataset terms before shipping
    run_dir = Path(os.getenv("POLYVOICE_VITS_RUN", ROOT / "runs" / "vits_lj_local"))
    tokenizer_path = ROOT / "data" / "datasets" / "ljspeech" / "tokenizer.json"

    def __init__(self):
        self.device = detect_device(); self.net = None; self.tok = None; self.step = None

    def _latest(self):
        cks = sorted(glob.glob(str(self.run_dir / "ckpt" / "step_*.pt")))
        return cks[-1] if cks else None

    def is_available(self):
        if not self._latest(): return False, "no trained checkpoint found (run scripts/train_vits.py)"
        if not self.tokenizer_path.exists(): return False, "tokenizer.json missing (run scripts/preprocess_data.py)"
        return True, ""

    def supported_languages(self): return ["en"]
    def supported_features(self):
        return {"speaking_rate": True, "voice_cloning": False, "emotion": False, "streaming": False, "multi_voice": False}

    def load_model(self, language=None):
        ck = self._latest()
        if self.net is not None and self.step == self._step_of(ck): return
        ok, why = self.is_available()
        if not ok: raise RuntimeError(why)
        s = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = VitsConfig(**_tupleize(s["model_cfg"]))
        net = VITS(cfg); net.load_state_dict(s["G"]); net.eval()
        try: net.to(self.device)
        except Exception: self.device = torch.device("cpu"); net.to(self.device)
        self.net, self.tok, self.step, self.cfg = net, CharTokenizer.load(self.tokenizer_path), s["step"], cfg

    @staticmethod
    def _step_of(p): return int(Path(p).stem.split("_")[1]) if p else None

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.load_model()
        ids = self.tok.encode(req.text, "en")
        if len(ids) <= 1: raise ValueError("Text is empty after normalization")
        x = torch.tensor([ids], device=self.device); xl = torch.tensor([len(ids)], device=self.device)
        zero = torch.zeros(1, dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        try: w, _, _ = self.net.infer(x, xl, sid=zero, lid=zero, length_scale=1.0 / req.speaking_rate)
        except RuntimeError:   # unsupported MPS op / memory: retry on CPU
            self.net.to("cpu"); self.device = torch.device("cpu")
            w, _, _ = self.net.infer(x.cpu(), xl.cpu(), sid=zero.cpu(), lid=zero.cpu(), length_scale=1.0 / req.speaking_rate)
        audio = w[0, 0].float().cpu().numpy(); dt = time.perf_counter() - t0
        sr = self.cfg.sample_rate
        return SynthesisResult(audio, sr, self.model_id, {"device": str(self.device), "gen_seconds": dt, "rtf": dt / (len(audio) / sr), "train_step": self.step})

    def estimate_resource_usage(self, language=None):
        return {"approx_params_millions": 10.8, "approx_disk_mb": 45, "note": "single speaker (LJSpeech), character input, 16 kHz"}

    def release_model(self):
        self.net = None
        if self.device.type == "mps": torch.mps.empty_cache()
