"""F5-TTS worker (runs in .venv-f5). JSON line in -> JSON line out. Weights: CC-BY-NC-4.0 (research only)."""
import json, os, sys, time
_proto = sys.stdout; sys.stdout = sys.stderr  # library prints must not corrupt the JSON protocol channel
def send(o): print(json.dumps(o), file=_proto, flush=True)

# torchaudio>=2.9 routes load() through torchcodec, which needs shared FFmpeg libs (absent here). Use soundfile instead.
import torch, torchaudio, soundfile as _sf
def _load(path, *a, **k):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr
torchaudio.load = _load
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import soundfile as sf, torch
from f5_tts.api import F5TTS

dev = os.environ.get("POLYVOICE_F5_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def build(d): return F5TTS(model="F5TTS_v1_Base", device=d)
try: tts = build(dev)
except Exception as e:
    print(f"[f5 worker] {dev} init failed ({e}); using cpu", file=sys.stderr); dev = "cpu"; tts = build(dev)
send({"ready": True, "device": dev})

for line in sys.stdin:
    req = json.loads(line)
    try:
        t0 = time.perf_counter()
        wav, sr, _ = tts.infer(ref_file=req["reference_audio"], ref_text=req["reference_text"], gen_text=req["text"], speed=req.get("speed", 1.0))
        sf.write(req["out_path"], wav, sr); dt = time.perf_counter() - t0
        send({"ok": True, "gen_seconds": dt, "rtf": dt / (len(wav) / sr), "device": dev})
    except Exception as e:
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
