"""XTTS-v2 worker: runs inside .venv-xtts. Protocol: one JSON request per stdin line -> one JSON reply per stdout line."""
import json, os, sys, time
_proto = sys.stdout; sys.stdout = sys.stderr  # library prints must not corrupt the JSON protocol channel
def send(o): print(json.dumps(o), file=_proto, flush=True)

# torchaudio>=2.9 routes load() through torchcodec, which needs shared FFmpeg libs (absent here). Use soundfile instead.
import torch, torchaudio, soundfile as _sf
def _load(path, *a, **k):
    data, sr = _sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T.copy()), sr
torchaudio.load = _load
os.environ["COQUI_TOS_AGREED"] = "1"  # set only after the developer explicitly accepted the CPML (see docs/model_licenses.md)
import torch, soundfile as sf
from TTS.api import TTS

dev = os.environ.get("POLYVOICE_XTTS_DEVICE", "cpu")  # MPS support in XTTS is unreliable; CPU is the verified default
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
speakers = sorted(tts.synthesizer.tts_model.speaker_manager.speakers.keys())
send({"ready": True, "device": dev, "speakers": speakers})

for line in sys.stdin:
    req = json.loads(line)
    try:
        t0 = time.perf_counter()
        kw = {"speaker_wav": req["reference_audio"]} if req.get("reference_audio") else {"speaker": req["speaker"]}
        wav = tts.tts(text=req["text"], language=req["language"], speed=req.get("speed", 1.0), **kw)
        sf.write(req["out_path"], wav, 24000)
        dt = time.perf_counter() - t0
        send({"ok": True, "seconds": len(wav) / 24000, "gen_seconds": dt, "rtf": dt / (len(wav) / 24000)})
    except Exception as e:
        send({"ok": False, "error": f"{type(e).__name__}: {e}"})
