"""Meta MMS-TTS adapter (VITS checkpoints, one per language). CC-BY-NC-4.0 weights."""
from __future__ import annotations
import time
import numpy as np, torch
from transformers import VitsModel, AutoTokenizer
from ml.models.base import TTSBackend, SynthesisRequest, SynthesisResult
from ml.inference.device import detect_device
from ml.preprocessing.text_norm import normalize_text

# ISO 639-1 -> MMS ISO 639-3 checkpoint suffix. Only entries verified to exist on the Hub.
MMS_LANGS: dict[str, str] = {
    "en": "eng", "hi": "hin", "es": "spa", "fr": "fra", "de": "deu",
    "ar": "ara", "ko": "kor", "pt": "por",
}  # verified via HfApi.model_info; ja (jpn) and it (ita) have NO MMS-TTS checkpoint

class MMSBackend(TTSBackend):
    model_id = "mms-tts"
    display_name = "Meta MMS-TTS"
    license = "CC-BY-NC-4.0 (weights)"
    commercial_use = False

    def __init__(self, device: str | None = None):
        self.device = detect_device(device)
        self._models: dict[str, tuple[VitsModel, AutoTokenizer]] = {}

    def supported_languages(self): return list(MMS_LANGS)
    def supported_features(self):
        return {"speaking_rate": True, "voice_cloning": False, "emotion": False,
                "streaming": False, "multi_voice": False}

    def load_model(self, language: str | None = None):
        language = language or "en"
        if language not in MMS_LANGS:
            raise ValueError(f"MMS backend has no verified checkpoint for '{language}'")
        if language in self._models:
            return
        repo = f"facebook/mms-tts-{MMS_LANGS[language]}"
        tok = AutoTokenizer.from_pretrained(repo)
        model = VitsModel.from_pretrained(repo).eval()
        try:
            model.to(self.device)
        except Exception:  # MPS op/memory problem: fall back to CPU explicitly
            self.device = torch.device("cpu"); model.to(self.device)
        self._models[language] = (model, tok)

    def synthesize(self, req: SynthesisRequest) -> SynthesisResult:
        self.load_model(req.language)
        model, tok = self._models[req.language]
        text = normalize_text(req.text, req.language)
        if not text:
            raise ValueError("Text is empty after normalization")
        if getattr(tok, "is_uroman", False):
            import uroman as ur
            text = ur.Uroman().romanize_string(text).lower()
        model.speaking_rate = float(req.speaking_rate)
        inputs = tok(text, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            try:
                wav = model(**inputs).waveform
            except RuntimeError:  # unsupported MPS op: retry on CPU
                model.to("cpu"); self.device = torch.device("cpu")
                wav = model(**{k: v.to("cpu") for k, v in inputs.items()}).waveform
        wav = wav[0].float().cpu().numpy()
        dt = time.perf_counter() - t0
        sr = model.config.sampling_rate
        return SynthesisResult(wav, sr, self.model_id,
                               {"device": str(self.device), "gen_seconds": dt,
                                "rtf": dt / (len(wav) / sr)})

    def estimate_resource_usage(self, language=None):
        return {"approx_params_millions": 36, "approx_disk_mb": 145, "note": "per language checkpoint"}

    def release_model(self):
        self._models.clear()
        if self.device.type == "mps": torch.mps.empty_cache()
        elif self.device.type == "cuda": torch.cuda.empty_cache()
