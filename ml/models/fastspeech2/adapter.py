"""FastSpeech 2 backend. TRAINED FROM SCRATCH here: the acoustic model (text -> 24 kHz/100-bin log-mel).
PRETRAINED, NOT ours: the Vocos mel vocoder (charactr/vocos-mel-24khz, MIT). Duration labels came from wav2vec2-base-960h CTC alignment."""
from __future__ import annotations
import glob, os, time
from pathlib import Path
import torch
from ml.models.base import TTSBackend, SynthesisRequest, SynthesisResult
from ml.inference.device import detect_device
from ml.models.fastspeech2.model import FastSpeech2, FS2Config
from ml.preprocessing.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[3]

class FastSpeech2Backend(TTSBackend):
    model_id = "polyvoice-fastspeech2-lj"
    display_name = "PolyVoice FastSpeech 2 (custom, LJSpeech)"
    license = "Acoustic model: own weights (LJSpeech, public domain per dataset page). Vocoder: Vocos, MIT (pretrained)"
    commercial_use = True
    run_dir = Path(os.getenv("POLYVOICE_FS2_RUN", ROOT / "runs" / "fs2_lj_local"))
    tokenizer_path = ROOT / "data" / "datasets" / "ljspeech" / "tokenizer.json"

    def __init__(self):
        self.device = detect_device(); self.net = None; self.vocos = None; self.step = None

    def _latest(self):
        c = sorted(glob.glob(str(self.run_dir / "ckpt" / "step_*.pt"))); return c[-1] if c else None

    def is_available(self):
        if not self._latest(): return False, "no trained checkpoint found (run scripts/train_fastspeech2.py)"
        if not self.tokenizer_path.exists(): return False, "tokenizer.json missing"
        return True, ""

    def supported_languages(self): return ["en"]
    def supported_features(self):
        return {"speaking_rate": True, "voice_cloning": False, "emotion": False, "streaming": False, "multi_voice": False}

    def load_model(self, language=None):
        ck = self._latest(); ok, why = self.is_available()
        if not ok: raise RuntimeError(why)
        step = int(Path(ck).stem.split("_")[1])
        if self.net is not None and step == self.step: return
        s = torch.load(ck, map_location="cpu", weights_only=False)
        net = FastSpeech2(FS2Config(**s["model_cfg"])); net.load_state_dict(s["net"]); net.eval()
        try: net.to(self.device)
        except Exception: self.device = torch.device("cpu"); net.to(self.device)
        self.net, self.stats, self.step, self.tok = net, s["stats"], step, CharTokenizer.load(self.tokenizer_path)
        if self.vocos is None:
            from vocos import Vocos
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()      # CPU: istft is not reliably supported on MPS

    @torch.no_grad()
    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.load_model()
        ids = self.tok.encode(req.text, "en", add_blank=False)
        if not ids: raise ValueError("Text is empty after normalization")
        tokens = torch.tensor([[0] + ids + [0]], device=self.device); tl = torch.tensor([tokens.size(1)], device=self.device)
        t0 = time.perf_counter()
        try: o = self.net(tokens, tl, speed=req.speaking_rate)
        except RuntimeError:
            self.net.to("cpu"); self.device = torch.device("cpu"); o = self.net(tokens.cpu(), tl.cpu(), speed=req.speaking_rate)
        mel = (o["mel"].float().cpu() * self.stats["mel_std"] + self.stats["mel_mean"]).transpose(1, 2)    # [1,100,T]
        audio = self.vocos.decode(mel)[0].numpy(); dt = time.perf_counter() - t0
        return SynthesisResult(audio, 24000, self.model_id, {"device": str(self.device), "gen_seconds": dt, "rtf": dt / (len(audio) / 24000), "train_step": self.step})

    def estimate_resource_usage(self, language=None):
        return {"approx_params_millions": 14, "approx_disk_mb": 60, "note": "FS2 acoustic model (own) + Vocos vocoder (pretrained, ~54 MB)"}

    def release_model(self):
        self.net = None; self.vocos = None
        if self.device.type == "mps": torch.mps.empty_cache()
