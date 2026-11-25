"""Back-transcription intelligibility check: TTS audio -> Whisper -> CER vs. normalized reference."""
from __future__ import annotations
import re, unicodedata
import numpy as np, librosa

def cer(ref: str, hyp: str) -> float:
    r, h = list(ref), list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[-1, -1] / max(1, len(r))

def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    # Strip punctuation/symbols by Unicode category; \w would also drop combining vowel signs (Hindi etc.).
    s = "".join(c for c in s if unicodedata.category(c)[0] not in "PS")
    return re.sub(r"\s+", " ", s).strip()

class Transcriber:
    def __init__(self, model="openai/whisper-small", device="cpu"):
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=model, device=device)
    def transcribe(self, audio: np.ndarray, sr: int, language: str) -> str:
        a = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else audio
        return self.pipe({"raw": a, "sampling_rate": 16000}, generate_kwargs={"language": language, "task": "transcribe"})["text"]

def score(ref: str, hyp: str) -> float:
    return cer(_clean(ref), _clean(hyp))
