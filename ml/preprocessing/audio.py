"""Audio loading, validation, quality estimation and resampling."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np, soundfile as sf, librosa

class AudioRejected(ValueError): pass

@dataclass
class AudioReport:
    duration_s: float; sr_in: int; peak: float; rms_db: float; clip_frac: float; silence_frac: float; longest_silence_s: float

def load_validate(path, target_sr=16000, min_s=1.0, max_s=15.0, max_clip=0.001, max_silence_frac=0.6, max_silence_run_s=2.5):
    """Returns (float32 mono waveform at target_sr, AudioReport). Raises AudioRejected on corrupt/unusable audio."""
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioRejected(f"unreadable: {e}")
    y = y.mean(axis=1)                                   # mono
    if not np.isfinite(y).all(): raise AudioRejected("non-finite samples")
    dur = len(y) / sr
    if dur < min_s: raise AudioRejected(f"too short ({dur:.2f}s)")
    if dur > max_s: raise AudioRejected(f"too long ({dur:.2f}s)")
    peak = float(np.abs(y).max())
    if peak < 1e-4: raise AudioRejected("silent file")
    clip = float((np.abs(y) >= 0.999).mean())
    if clip > max_clip: raise AudioRejected(f"clipping {clip:.4f}")
    if sr != target_sr: y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    hop = int(0.02 * target_sr)
    frames = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    thr = frames.max() * 10 ** (-40 / 20)                # 40 dB below peak frame
    quiet = frames < thr
    run = best = 0
    for q in quiet:
        run = run + 1 if q else 0; best = max(best, run)
    sil_frac, sil_run = float(quiet.mean()), best * 0.02
    if sil_frac > max_silence_frac: raise AudioRejected(f"mostly silence ({sil_frac:.2f})")
    if sil_run > max_silence_run_s: raise AudioRejected(f"long silence ({sil_run:.2f}s)")
    rms_db = float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9))
    return y.astype(np.float32), AudioReport(len(y) / target_sr, sr, peak, rms_db, clip, sil_frac, sil_run)

def to_int16(y): return (np.clip(y, -1, 1) * 32767).astype(np.int16)
