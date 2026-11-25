"""FastSpeech 2 targets: Vocos-compatible 24 kHz/100-bin log-mel, frame-level F0 (YIN) and energy.
Durations come separately from teacher alignments (scripts/extract_durations.py)."""
from __future__ import annotations
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np, soundfile as sf, librosa, torch

SR, HOP, N_MELS = 24000, 256, 100
FMIN, FMAX = 65.0, 400.0   # LJSpeech is a single female speaker; YIN range chosen to cover her F0 with margin
_fe = None

def _extractor():
    global _fe
    if _fe is None:
        from vocos.feature_extractors import MelSpectrogramFeatures
        _fe = MelSpectrogramFeatures(sample_rate=SR, n_fft=1024, hop_length=HOP, n_mels=N_MELS, padding="center")
    return _fe

def extract(wav_path, fmin=FMIN, fmax=FMAX) -> dict:
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=True); y = y.mean(1)
    return extract_array(y, sr, fmin, fmax)

def extract_array(y, sr, fmin=FMIN, fmax=FMAX) -> dict:
    """Same features from an in-memory mono waveform. fmin/fmax set the YIN range (multi-speaker data uses a wider range)."""
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR, res_type="soxr_hq")
    with torch.no_grad(): mel = _extractor()(torch.from_numpy(y)[None])[0].numpy()          # [100, T]
    T = mel.shape[1]
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=SR, frame_length=2048, hop_length=HOP)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP, center=True)[0]
    n = min(T, len(f0), len(rms)); f0, rms = np.pad(f0[:n], (0, T - n), mode="edge"), np.pad(rms[:n], (0, T - n), mode="edge")
    voiced = (rms > rms.max() * 10 ** (-35 / 20)) & (f0 > fmin * 1.02) & (f0 < fmax * 0.98)   # YIN pins at bounds when unvoiced
    f0 = np.where(voiced, f0, 0.0)
    return {"mel": mel.astype(np.float16), "f0": f0.astype(np.float32), "energy": rms.astype(np.float32)}

def _work(a):
    uid, wav, out = a
    d = extract(wav); np.savez(out / f"{uid}.npz", **d); return uid, d["mel"].shape[1]

def build(raw_wavs: Path, manifest: Path, out: Path, workers=4):
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in manifest.read_text().splitlines()]
    todo = [(r["id"], raw_wavs / f"{r['id']}.wav", out) for r in rows if not (out / f"{r['id']}.npz").exists()]
    with Pool(workers) as p: res = p.map(_work, todo, chunksize=32)
    return {"computed": len(res), "total": len(rows)}


def _work_npy(a):
    uid, npy, out, fmin, fmax = a
    y = np.load(npy).astype(np.float32) / 32767                # cached 16 kHz int16 audio -> resampled to 24 kHz in extract_array
    d = extract_array(y, 16000, fmin, fmax); np.savez(out / f"{uid}.npz", **d); return uid, d["mel"].shape[1]

def build_from_cache(lang_dir: Path, workers=4, fmin=60.0, fmax=500.0):
    """Multilingual layout: <lang_dir>/wavs16k/*.npy + manifest.jsonl -> <lang_dir>/fs2/*.npz. NOTE: source audio is 16 kHz, so mel bins above ~8 kHz are empty
    (the vocoder will produce band-limited 24 kHz output)."""
    out = lang_dir / "fs2"; out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in (lang_dir / "manifest.jsonl").read_text().splitlines()]
    todo = [(r["id"], lang_dir / "wavs16k" / f"{r['id']}.npy", out, fmin, fmax) for r in rows if not (out / f"{r['id']}.npz").exists()]
    with Pool(workers) as p: res = p.map(_work_npy, todo, chunksize=32)
    return {"lang": lang_dir.name, "computed": len(res), "total": len(rows)}
